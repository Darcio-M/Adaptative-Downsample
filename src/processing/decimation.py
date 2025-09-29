import time
import math
import logging
import pymeshlab
from typing import List, Tuple, Optional
from pathlib import Path

from src import config

def _decimate_mesh_adaptive(
    in_ply: str, out_ply: str, target_faces: int
) -> dict:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_ply)
    
    kwargs = {
        'targetfacenum': target_faces,
        'qualitythr': 0.3,
        'qualityweight': True,
        'preserveboundary': False,
        'preservetopology': False,
        'optimalplacement': True,
        'planarquadric': False,
        'autoclean': True
    }
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', **kwargs)
    ms.save_current_mesh(out_ply)
    
    return {'out': out_ply, 'faces': ms.current_mesh().face_number()}

def run_adaptive_decimation_pipeline() -> List[Tuple[str, float, str, Optional[str]]]:
    entries = []
    input_ply = config.CURVATURE_PLY_PATH
    
    for lod in config.ADAPTIVE_LODS:
        out_name = f"{config.ADAPTIVE_PREFIX}_{lod}.ply"
        out_path = str(Path(config.OUTPUT_DIR) / out_name)
        
        start = time.perf_counter()
        status = "ok"
        result = None
        try:
            res_dict = _decimate_mesh_adaptive(input_ply, out_path, lod)
            result = f"Arquivo salvo com {res_dict['faces']} faces (alvo {lod})"
            logging.info("Dizimação adaptativa concluída: %s", result)
        except Exception as e:
            status = f"erro: {e}"
            logging.exception("Falha na dizimação adaptativa para %s: %s", out_name, e)
        elapsed = time.perf_counter() - start
        entries.append((f"adaptive_{out_name}", elapsed, status, result))
        
    return entries

def _estimate_edge_length(area: float, target_faces: int) -> float:
    avg_tri_area = area / float(target_faces)
    return math.sqrt(4.0 * avg_tri_area / math.sqrt(3.0))

def run_uniform_remeshing_pipeline() -> List[Tuple[str, float, str, Optional[str]]]:
    results = []
    input_ply = config.CURVATURE_PLY_PATH

    ms_area = pymeshlab.MeshSet()
    ms_area.load_new_mesh(input_ply)
    total_area = ms_area.get_geometric_measures().get('surface_area', 0.0)

    if total_area <= 0:
        raise ValueError("Área da malha de entrada é zero. Impossível continuar.")

    logging.info("Área de superfície detectada para remeshing: %.6f", total_area)

    for target_faces in sorted(config.UNIFORM_LODS, reverse=True):
        out_name = f"{config.UNIFORM_PREFIX}_{target_faces}.ply"
        out_path = str(Path(config.OUTPUT_DIR) / out_name)
        
        start = time.perf_counter()
        status = "ok"
        result_msg = None

        try:
            target_edge_len = _estimate_edge_length(total_area, target_faces)
            logging.info("LOD Uniforme %d: edge_len alvo = %.6f", target_faces, target_edge_len)

            ms_proc = pymeshlab.MeshSet()
            ms_proc.load_new_mesh(input_ply)
            ms_proc.apply_filter(
                'meshing_isotropic_explicit_remeshing',
                iterations=config.UNIFORM_ITERATIONS,
                adaptive=config.UNIFORM_ADAPTIVE,
                targetlen=pymeshlab.PureValue(target_edge_len),
                featuredeg=config.UNIFORM_FEATURE_ANGLE
            )
            ms_proc.save_current_mesh(out_path)
            
            final_faces = ms_proc.current_mesh().face_number()
            result_msg = f"Arquivo salvo com {final_faces} faces (alvo {target_faces})"
            logging.info("Remesh uniforme concluído e salvo: %s", out_path)

        except Exception as e:
            status = f"erro: {e}"
            logging.exception("Erro no remeshing uniforme para LOD %d: %s", target_faces, e)

        elapsed = time.perf_counter() - start
        results.append((f"uniform_{out_name}", elapsed, status, result_msg))

    return results