from logging import Logger

from src.utils.Singleton import Singleton
from src.utils.logger.MPLogger import MPLogger


class LoggerManager(metaclass=Singleton):
    def __init__(self):
        self.logger = MPLogger(log_name="Payload_Analyzer", log_level="DEBUG", log_dir=None).get_logger()
        self.logger.info(f"Logger initialized...")

    @staticmethod
    def get() -> Logger:
        return LoggerManager().logger