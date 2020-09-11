from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__("my_objects/bread.xml")


class TorsoObject(MujocoXMLObject):

    def __init__(self):
        super().__init__("my_objects/torso.xml")
