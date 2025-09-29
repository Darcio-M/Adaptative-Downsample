from pathlib import Path

# --- Diretórios Base ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "malhas_entrada"
OUTPUT_DIR = BASE_DIR / "malhas_saida"

# --- Nomes de Arquivos ---
INPUT_FILE_NAME = "leaoaa2Init.ply"
CURVATURE_OUTPUT_NAME = "curv_out.ply"
TIMINGS_FILE = "timings.txt"

INPUT_PLY_PATH = str(INPUT_DIR / INPUT_FILE_NAME)
CURVATURE_PLY_PATH = str(OUTPUT_DIR / CURVATURE_OUTPUT_NAME)

# --- Parâmetros de Curvatura ---
CURV_TARGET_PERC = 0.15
CURV_KNN = 3
CURV_SMOOTH_ITERS = 6
CURV_SMOOTH_LAMBDA = 1.0
CURV_LOWER_PCT = 2.0
CURV_UPPER_PCT = 98.0

# --- Parâmetros de Dizimação Adaptativa ---
ADAPTIVE_LODS = [100000, 80000, 60000]
ADAPTIVE_TARGET_PERC = 0.0
ADAPTIVE_PREFIX = "adaptive_LOD"

# --- Parâmetros de Remeshing Uniforme ---
UNIFORM_LODS = [100000, 80000, 60000]
UNIFORM_PREFIX = "uniform_LOD"
UNIFORM_ITERATIONS = 10
UNIFORM_ADAPTIVE = False
UNIFORM_FEATURE_ANGLE = 30.0