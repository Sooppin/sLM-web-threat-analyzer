import os

from src.utils.Singleton import Singleton
from src.utils.FileUtils import FileUtils
from src.utils.ConfigUtils import ConfigUtils


class ConfigManager(metaclass=Singleton):
    def __init__(self):
        conf_filename = f"{os.getcwd()}/conf/conf.xml"
        # Dev
        if not FileUtils.is_exist(conf_filename):
            conf_filename = f"{FileUtils.get_realpath(__file__)}/../../conf/conf.xml"
        self.conf = ConfigUtils.load_conf_xml(conf_filename)

    def get(self, key, default=None) -> object:
        return self.conf.get(key, default)
