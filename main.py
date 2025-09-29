import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

from src import config
from src.processing import curvature, decimation
from src.utils.timings import write_timings_file

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    Path(config.INPUT_DIR).mkdir(exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

    run_start_dt = datetime.utcnow()
    run_start_ts = time.perf_counter()
    logs: List[Tuple[str, float, str, Optional[str]]] = []

    logging.info("Iniciando o pipeline de processamento de malha...")

    # --- 1) Cálculo de Curvatura ---
    logging.info("Etapa 1: Calculando a curvatura...")
    try:
        curv_elapsed, curv_status, curv_result = curvature.run_curvature_step()
        logs.append(("curvature", curv_elapsed, curv_status, curv_result))
        if curv_status != "ok":
            raise RuntimeError(f"Falha na etapa de curvatura: {curv_result}")
        logging.info("Etapa de curvatura concluída com sucesso.")
    except Exception as e:
        logging.exception("Erro fatal na etapa de curvatura. Abortando.")
        logs.append(("curvature", time.perf_counter() - run_start_ts, f"erro fatal: {e}", None))
        return

    # --- 2) Dizimação Adaptativa ---
    logging.info("Etapa 2: Executando a dizimação adaptativa...")
    try:
        adaptive_entries = decimation.run_adaptive_decimation_pipeline()
        adaptive_total_elapsed = sum(e[1] for e in adaptive_entries)
        logs.append(("adaptive_decimation_total", adaptive_total_elapsed, "ok", None))
        logs.extend(adaptive_entries)
        logging.info("Dizimação adaptativa concluída.")
    except Exception as e:
        elapsed = time.perf_counter() - run_start_ts
        logging.exception("Erro fatal durante a dizimação adaptativa: %s", e)
        logs.append(("adaptive_decimation_total", elapsed, f"erro fatal: {e}", None))


    # --- 3) Remeshing Uniforme ---
    logging.info("Etapa 3: Executando o remeshing uniforme...")
    try:
        uniform_entries = decimation.run_uniform_remeshing_pipeline()
        uniform_total_elapsed = sum(e[1] for e in uniform_entries)
        logs.append(("uniform_remeshing_total", uniform_total_elapsed, "ok", None))
        logs.extend(uniform_entries)
        logging.info("Remeshing uniforme concluído.")
    except Exception as e:
        elapsed = time.perf_counter() - run_start_ts
        logging.exception("Erro fatal ao gerar LODs uniformes: %s", e)
        logs.append(("uniform_remeshing_total", elapsed, f"erro fatal: {e}", None))


    # --- Finalização ---
    run_end_dt = datetime.utcnow()
    total_elapsed = time.perf_counter() - run_start_ts

    # Salvar arquivo de tempos
    timings_path = Path(config.OUTPUT_DIR) / config.TIMINGS_FILE
    write_timings_file(str(timings_path), run_start_dt, run_end_dt, total_elapsed, logs)
    logging.info(f"Pipeline concluído em {total_elapsed:.2f} segundos.")


if __name__ == "__main__":
    main()