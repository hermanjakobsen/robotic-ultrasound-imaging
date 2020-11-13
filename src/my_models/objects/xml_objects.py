import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

class TorsoObject(MujocoXMLObject):
    """
    Torso object
    """

    def __init__(self):
        super().__init__("my_models/assets/objects/human_torso.xml")


class SoftTorsoObject(MujocoXMLObject):
    """
    Soft object
    """

    def __init__(self, damping=None, stiffness=None):
        super().__init__("my_models/assets/objects/soft_human_torso.xml")

        self.damping = damping
        self.stiffness = stiffness

        if self.damping is not None:
            self._set_damping(damping)
        if self.stiffness is not None:
            self._set_stiffness(stiffness)


    def _get_composite_element(self):
        collision = self.worldbody.find("./body/body[@name='collision']")
        return collision.find("./composite[@name='soft_body']")


    def _set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))


    def _set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))


class BoxObject(MujocoXMLObject):
    """
    Box object
    """

    def __init__(self):
        super().__init__("my_models/assets/objects/box.xml")