import os
import time
import logging
import numpy as np
import pymeshlab
import matplotlib
from plyfile import PlyElement, PlyData
from typing import Tuple, Optional
from src import config

def _try_kdtree_backends():
    try:
        from scipy.spatial import cKDTree as KDTree
        return 'scipy', KDTree
    except ImportError:
        pass
    try:
        import open3d as o3d
        return 'open3d', o3d
    except ImportError:
        pass
    return None, None

def _transfer_scalars_knn(
    small_verts: np.ndarray,
    small_vals: np.ndarray,
    large_verts: np.ndarray,
    k: int = 3,
    use_weights: bool = True
) -> np.ndarray:
    backend, lib = _try_kdtree_backends()
    if backend == 'scipy':
        KDTree = lib
        tree = KDTree(small_verts)
        dists, idxs = tree.query(large_verts, k=min(k, small_verts.shape[0]))
        if dists.ndim == 1:
            dists = dists[:, None]; idxs = idxs[:, None]
        if use_weights:
            eps = 1e-12
            weights = 1.0 / (dists + eps)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            safe_weights_sum = np.where(weights_sum == 0, 1, weights_sum)
            vals = np.sum(weights * small_vals[idxs], axis=1) / safe_weights_sum[:,0]
        else:
            vals = small_vals[idxs[:,0]]
        return vals
    elif backend == 'open3d':
        o3d = lib
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(small_verts)
        kdt = o3d.geometry.KDTreeFlann(pcd)
        N = large_verts.shape[0]
        vals = np.empty(N, dtype=np.float64)
        for i in range(N):
            _, idx, d = kdt.search_knn_vector_3d(large_verts[i], min(k, small_verts.shape[0]))
            idx = np.array(idx, dtype=np.int64)
            d = np.array(d, dtype=np.float64)
            if use_weights and d.size > 0:
                weights = 1.0 / (np.sqrt(d) + 1e-12)
                if weights.sum() > 0:
                    vals[i] = (weights * small_vals[idx]).sum() / weights.sum()
                else:
                    vals[i] = small_vals[idx[0]]
            else:
                vals[i] = small_vals[idx[0]]
        return vals
    else:
        N = large_verts.shape[0]
        vals = np.empty(N, dtype=np.float64)
        for i in range(N):
            dists = np.linalg.norm(small_verts - large_verts[i], axis=1)
            vals[i] = small_vals[np.argmin(dists)]
        return vals

def _build_vertex_neighbors_and_cot_weights(verts: np.ndarray, faces: np.ndarray):
    n = verts.shape[0]
    neigh = [set() for _ in range(n)]
    from collections import defaultdict
    w = defaultdict(float)
    def cot(a, b):
        dot_product = np.dot(a, b)
        cross_product_norm = np.linalg.norm(np.cross(a, b))
        return dot_product / cross_product_norm if cross_product_norm != 0 else 0.0
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = verts[i], verts[j], verts[k]
        ci = cot(vj - vi, vk - vi); cj = cot(vi - vj, vk - vj); ck = cot(vi - vk, vj - vk)
        w[(j, k)] += ci; w[(k, j)] += ci
        w[(i, k)] += cj; w[(k, i)] += cj
        w[(i, j)] += ck; w[(j, i)] += ck
        neigh[i].update((j, k)); neigh[j].update((i, k)); neigh[k].update((i, j))
    nbrs = [np.array(list(n_set), dtype=np.int64) for n_set in neigh]
    nbrs_w = [np.array([w.get((i, int(j)), 0.0) for j in n_list], dtype=np.float64) for i, n_list in enumerate(nbrs)]
    return nbrs, nbrs_w

def _smooth_scalar_cotangent_jacobi(scalar: np.ndarray, neighbors, weights, iterations: int = 10, lamb: float = 1.0):
    v = scalar.astype(np.float64).copy()
    for _ in range(iterations):
        new_v = v.copy()
        for i in range(len(v)):
            nbr = neighbors[i]
            if len(nbr) > 0:
                w = weights[i]
                sum_w = np.sum(w)
                if sum_w > 0:
                    avg = np.sum(w * v[nbr]) / sum_w
                    new_v[i] = (1.0 - lamb) * v[i] + lamb * avg
        v = new_v
    return v

def _get_mesh_arrays(ms: pymeshlab.MeshSet) -> Tuple[np.ndarray, np.ndarray]:
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()

def _normalize_symmetric_percentile(v: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    lo = np.percentile(v, lower_pct)
    hi = np.percentile(v, upper_pct)
    v_abs_max = max(abs(lo), abs(hi))
    if v_abs_max == 0: return np.zeros_like(v)
    v_clipped = np.clip(v, -v_abs_max, v_abs_max)
    return (v_clipped + v_abs_max) / (2.0 * v_abs_max)

def _colormap_from_norm(snorm: np.ndarray, cmap_name: str = 'coolwarm') -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgbf = cmap(snorm)[:, :3]
    return (rgbf * 255.0).astype(np.uint8)

def _write_ply_from_arrays(verts: np.ndarray, faces: np.ndarray, quality: np.ndarray, rgb: np.ndarray, out_ply: str):
    vert_arr = np.zeros(verts.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vert_arr['x'], vert_arr['y'], vert_arr['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    vert_arr['quality'] = quality
    vert_arr['red'], vert_arr['green'], vert_arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    face_arr = np.empty(faces.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    face_arr['vertex_indices'] = faces
    vert_el = PlyElement.describe(vert_arr, 'vertex')
    face_el = PlyElement.describe(face_arr, 'face')
    PlyData([vert_el, face_el], text=True).write(out_ply)

def _get_curvature_scalars_from_mesh(ms: pymeshlab.MeshSet) -> np.ndarray:
    m = ms.current_mesh()
    candidate_methods = [
        'vertex_scalar_array',
        'vertex_quality_array',
    ]
    for method_name in candidate_methods:
        if hasattr(m, method_name):
            try:
                scalar_array = getattr(m, method_name)()
                return np.asarray(scalar_array).reshape(-1)
            except Exception as e:
                logging.warning("Método '%s' encontrado, mas falhou ao ser chamado: %s", method_name, e)
    
    raise RuntimeError("Não foi possível ler o array de escalares da malha após o cálculo da curvatura.")

def _process_curvature_pipeline(in_ply: str, out_ply_after: str, target_perc: float, knn_k: int, use_distance_weights: bool, smooth_iterations: int, smooth_lambda: float, lower_pct: float, upper_pct: float) -> str:
    if not os.path.isfile(in_ply):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {in_ply}")
    
    ms_dense = pymeshlab.MeshSet()
    ms_dense.load_new_mesh(in_ply)
    verts_dense, faces_dense = _get_mesh_arrays(ms_dense)

    ms_simp = pymeshlab.MeshSet()
    ms_simp.add_mesh(ms_dense.current_mesh(), "base_mesh")
    ms_simp.apply_filter('meshing_decimation_quadric_edge_collapse', targetperc=target_perc, preservetopology=False, autoclean=True)
    
    try:
        logging.info("Iniciando reparo da malha simplificada...")
        ms_simp.apply_filter('meshing_remove_unreferenced_vertices')
        ms_simp.apply_filter('meshing_repair_non_manifold_edges')
        ms_simp.apply_filter('meshing_remove_duplicate_faces')
        logging.info("Reparo concluído.")
    except Exception as e:
        logging.warning("Ocorreu um erro durante a etapa de reparo da malha, o que pode ser normal: %s", e)
        pass 
    
    ms_simp.apply_filter('compute_curvature_principal_directions_per_vertex', method=3, curvcolormethod=0)
    
    v_scalar_simp = _get_curvature_scalars_from_mesh(ms_simp)
    verts_simp = ms_simp.current_mesh().vertex_matrix()

    vals_transferred = _transfer_scalars_knn(verts_simp, v_scalar_simp, verts_dense, k=knn_k, use_weights=use_distance_weights)

    neighbors_dense, weights_dense = _build_vertex_neighbors_and_cot_weights(verts_dense, faces_dense)
    v_smoothed_dense = _smooth_scalar_cotangent_jacobi(vals_transferred, neighbors_dense, weights_dense, iterations=smooth_iterations, lamb=smooth_lambda)
    
    snorm = _normalize_symmetric_percentile(v_smoothed_dense, lower_pct=lower_pct, upper_pct=upper_pct)
    rgb = _colormap_from_norm(snorm, cmap_name='coolwarm')

    _write_ply_from_arrays(verts_dense, faces_dense, v_smoothed_dense, rgb, out_ply_after)
    
    return out_ply_after

def run_curvature_step() -> Tuple[float, str, Optional[str]]:
    start = time.perf_counter()
    status = "ok"
    result = None
    try:
        result = _process_curvature_pipeline(
            in_ply=config.INPUT_PLY_PATH,
            out_ply_after=config.CURVATURE_PLY_PATH,
            target_perc=config.CURV_TARGET_PERC,
            knn_k=config.CURV_KNN,
            use_distance_weights=True,
            smooth_iterations=config.CURV_SMOOTH_ITERS,
            smooth_lambda=config.CURV_SMOOTH_LAMBDA,
            lower_pct=config.CURV_LOWER_PCT,
            upper_pct=config.CURV_UPPER_PCT
        )
        logging.info("Processo de curvatura finalizado. Saída: %s", result)
    except Exception as e:
        status = f"erro: {e}"
        result = None
        logging.exception("Erro durante o processamento de curvatura: %s", e)
    
    elapsed = time.perf_counter() - start
    return elapsed, status, result