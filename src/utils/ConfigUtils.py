from src.utils.XMLUtils import XMLUtils


class ConfigUtils:
    @staticmethod
    def load_conf_xml(filename):
        conf_dict = {}

        conf_tree = XMLUtils.xml_load(filename=filename)
        configuration = conf_tree.getroot()
        properties = configuration.findall("property")

        for _property in properties:
            name = _property.findtext("name")
            value = _property.findtext("value")
            conf_dict[name] = value

        return conf_dict