import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class UltrasoundArena(Arena):
    """
    Workspace that contains an empty hospital bed.

    Args:
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        xml="my_models/assets/arenas/ultrasound_arena.xml",
    ):
        super().__init__(xml)
        self.bed_body = self.worldbody.find("./body[@name='bed']")
        self.bed_collision = self.bed_body.find("./geom[@name='bed_collision']")

    @property
    def bed_top_abs(self):
        """
        Grabs the absolute position of the bed top
        Returns:
            np.array: (x,y,z) bed position
        """
        return string_to_array(self.bed_body.get("pos"))