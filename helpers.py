import os
import pathlib
import sys

import loguru
from loguru import logger
import torch

stop_sign = """
                        ██                       
                      ██░░██                     
                    ██░░░░░░██                   
                  ██░░░░░░░░░░██                 
                  ██░░░░░░░░░░██                 
                ██░░░░░░░░░░░░░░██               
              ██░░░░░░██████░░░░░░██             
              ██░░░░░░██████░░░░░░██             
            ██░░░░░░░░██████░░░░░░░░██           
            ██░░░░░░░░██████░░░░░░░░██           
          ██░░░░░░░░░░██████░░░░░░░░░░██         
        ██░░░░░░░░░░░░██████░░░░░░░░░░░░██       
        ██░░░░░░░░░░░░██████░░░░░░░░░░░░██       
      ██░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░██     
      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██     
    ██░░░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░░░██   
    ██░░░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░░░██   
  ██░░░░░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░░░░░██ 
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██ 
    ██████████████████████████████████████████   
"""

logger.remove()
logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
log_level = "DEBUG"
logger.add(
    sys.stdout,
    level=log_level,
    format=logger_format,
    colorize=True,
    backtrace=True,
    diagnose=True,
)
logger.add(
    "app.log",
    level=log_level,
    format=logger_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
)


def is_exits(fp) -> bool:
    p = pathlib.Path(fp)
    return p.is_file()


def is_on_cpu() -> bool:
    logger.warning("RUNNING ON CPU!!!")
    print(stop_sign)
    # return os.getenv("USE_CPU") == "true"
    return torch.cuda.is_available()


def get_logger() -> loguru.logger:
    return logger
