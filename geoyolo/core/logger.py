import os
import re

import sys
import socket
import logging
from collections import deque
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

log_filename = (
    f"{"logs"}/inference-{os.path.basename(sys.prefix)}-{socket.gethostname()}.log"
)
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

filehandler = logging.FileHandler(filename=log_filename)
streamhandler = logging.StreamHandler(stream=open(os.devnull, "w", encoding="utf-8"))
formatter = logging.Formatter("%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")

filehandler.setFormatter(formatter)
streamhandler.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def show_log(n_lines, level="INFO", strict=False):
    """Show the most recent 'n_lines' of logs at or above the specified level.
    Designed specifically for inference_engine cli.
    Args:
        n_lines (int): max lines to search from end of log file
        level (str | int): log level as string or int (10-50:DEBUG-CRTITICAL)
        strict (bool): filter log lines of only specificed level up to 'n_lines'
            entries at that level
    Returns:
        None. Prints to console.

    """
    log_levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    min_level = log_levels.get(
        level.upper(), 20
    )  # if isinstance(level, str) else level
    pattern = re.compile(r"\[\s*(\w+)\s*\]")

    if not os.path.exists(log_filename):
        return None
    else:
        # min_level = log_levels.get(level.upper(), 20) if isinstance(level, str) else level
        with open(log_filename, "r") as f:
            lines = []
            for line in f:
                match = pattern.search(line)
                if match:
                    line_level = log_levels.get(match.group(1), 0)
                    if (strict and line_level == min_level) or (
                        not strict and line_level >= min_level
                    ):
                        lines.append(line)
            last_n_lines = deque(lines, maxlen=n_lines)

        for line in last_n_lines:
            print(line, end="")
        print(f"\nLog File: {log_filename}")
