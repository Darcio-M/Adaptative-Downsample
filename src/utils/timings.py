import logging
from datetime import datetime
from typing import List, Tuple, Optional

def write_timings_file(
    timings_path: str,
    run_start: datetime,
    run_end: datetime,
    total_elapsed: float,
    logs: List[Tuple[str, float, str, Optional[str]]]
) -> None:
    """Grava o arquivo de tempos de execução."""
    try:
        with open(timings_path, "w", encoding="utf-8") as f:
            f.write(f"Run start (UTC): {run_start.isoformat()}Z\n")
            f.write(f"Run end   (UTC): {run_end.isoformat()}Z\n")
            f.write(f"Total elapsed seconds: {total_elapsed:.6f}\n\n")
            for name, secs, status, result in logs:
                f.write(f"{name}: {secs:.6f} s | status: {status} | result: {result}\n")
        logging.info("Tempos de execução gravados em: %s", timings_path)
    except Exception as e:
        logging.exception("Falha ao gravar o arquivo de tempos: %s", e)