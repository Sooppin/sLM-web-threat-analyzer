from src.utils.Singleton import Singleton
from src.common.ConfigManager import ConfigManager


class Constants(metaclass=Singleton):
    __config_manager = ConfigManager()

    DIR_EMBEDDINGS_CACHE = f"{__config_manager.get('src.resources.embeddings_cache')}"
    DIR_MODEL = f"{__config_manager.get('src.resources.model')}"

    EMB_MODEL_NAME = __config_manager.get('embedded_model_name')
    EXT_MODEL_NAME = __config_manager.get('extract_model_name')
    EMB_MODEL_PATH = f"{DIR_MODEL}/{EMB_MODEL_NAME}"
    EXT_MODEL_PATH = f"{DIR_MODEL}/{EXT_MODEL_NAME}"

    DEVICE = __config_manager.get('device')
    BATCH_SIZE = int(__config_manager.get('batch_size'))
    THRESHOLD = float(__config_manager.get('threshold'))
    K = int(__config_manager.get('k'))

    PREFIX = __config_manager.get('prefix')
    MAX_LENGTH = int(__config_manager.get('max_length'))

    MAX_MEMORY_MB = int(__config_manager.get('max_memory_mb'))
