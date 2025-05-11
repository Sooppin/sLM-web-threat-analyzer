import torch
from logging import Logger

from src.common.LoggerManager import LoggerManager
from src.common.Constants import Constants


class CUDALimit:
    def __init__(self):
        self.logger: Logger = LoggerManager.get()
        self.max_memory_mb = Constants.MAX_MEMORY_MB
        self.device = Constants.DEVICE

    def set_memory_limit(self):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        total_memory_mb = total_memory / (1024**2)
        fraction = self.max_memory_mb / total_memory_mb
        fraction = min(fraction, 1.0)

        torch.cuda.set_per_process_memory_fraction(fraction, self.device)

        self.logger.info(f"GPU memory limit set to {self.max_memory_mb} MB ({fraction:.4f} fraction)")