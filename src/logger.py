"""
SAMPLE USAGE:

import logger
log = logger.get_logger("logger_name", logging.INFO, "[REPO]/logs", "YYYY-MM-DD-logger_name.log")
log.info("Logger has been created successfully")

"""

import logging
import logging.config
import os
from logging import Logger
from typing import Optional

LOADED_CFG = False
FORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


def get_logger(name: str = '',
               level: int = logging.INFO,
               output_dir: Optional[str] = None,
               log_file_name: Optional[str] = None) -> Logger:
    """Custom wrapper for getting logger."""

    name = name if name else __name__
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if output_dir and log_file_name and name:
        os.makedirs(output_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(output_dir, log_file_name), mode="a+", encoding='utf-8')
        formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
        handler.setFormatter(formatter)
        if len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[0])
        logger.addHandler(handler)
    else:
        logging.basicConfig(format=FORMAT, datefmt=DATEFMT, encoding='utf-8')
    return logger
