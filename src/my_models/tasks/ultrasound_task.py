import collections
from copy import deepcopy

from robosuite.models.world import MujocoWorldBase
from robosuite.models.tasks import UniformRandomSampler
from robosuite.models.objects import MujocoGeneratedObject, MujocoXMLObject
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class UltrasoundTask(MujocoWorldBase):
    """
    Creates MJCF model for a ultrasound task performed, usually on a table top (or similar surface).
    An ultrasound task consists of one robot holding an ultrasound probe and interacting with 
    a soft body in various settings. This class combines the robot, the table
    arena, and the objects into a single MJCF model.
    Args:
        mujoco_arena (Arena): MJCF model of robot workspace
        mujoco_robots (list of RobotModel): MJCF model of robot model(s) (list)
        mujoco_objects (OrderedDict of MujocoObject): a list of MJCF models of physical objects
        visual_objects (OrderedDict of MujocoObject): a list of MJCF models of visual-only objects that do not
            participate in collisions
        initializer (ObjectPositionSampler): placement sampler to initialize object positions.
    Raises:
        AssertionError: [Invalid input object type]
    """

    def __init__(
        self, 
        mujoco_arena, 
        mujoco_robots, 
        mujoco_objects_on_table,
        other_mujoco_objects=None, 
        visual_mujoco_objects_on_table=None,
        other_visual_mujoco_objects=None,
        initializer=None,
    ):
        super().__init__()

        self.merge_arena(mujoco_arena)
        for mujoco_robot in mujoco_robots:
            self.merge_robot(mujoco_robot)

        if initializer is None:
            initializer = UniformRandomSampler()

        if other_mujoco_objects is None:
            other_mujoco_objects = collections.OrderedDict()

        if visual_mujoco_objects_on_table is None:
            visual_mujoco_objects_on_table = collections.OrderedDict()

        if other_visual_mujoco_objects is None:
            other_visual_mujoco_objects = collections.OrderedDict()

        mujoco_objects_on_table = deepcopy(mujoco_objects_on_table)
        other_mujoco_objects = deepcopy(other_mujoco_objects)
        visual_mujoco_objects_on_table = deepcopy(visual_mujoco_objects_on_table)
        other_visual_mujoco_objects = deepcopy(other_visual_mujoco_objects)

        assert isinstance(mujoco_objects_on_table, collections.OrderedDict)
        assert isinstance(other_mujoco_objects, collections.OrderedDict)
        assert isinstance(visual_mujoco_objects_on_table, collections.OrderedDict)
        assert isinstance(other_visual_mujoco_objects, collections.OrderedDict)

        # xml manifestations of all objects
        self.objects_on_table = []
        self.other_objects = []
        self.visual_objects_on_table = []
        self.other_visual_objects = []

        self.merge_objects(
            mujoco_objects_on_table, 
            other_mujoco_objects, 
            visual_mujoco_objects_on_table, 
            other_visual_mujoco_objects
        )

        merged_objects_on_table = collections.OrderedDict(**mujoco_objects_on_table, **visual_mujoco_objects_on_table)

        self.mujoco_objects_on_table = mujoco_objects_on_table
        self.other_mujoco_objects = other_mujoco_objects 
        self.visual_mujoco_objects_on_table = visual_mujoco_objects_on_table
        self.other_visual_mujoco_objects = other_visual_mujoco_objects

        self.initializer = initializer
        self.initializer.setup(merged_objects_on_table, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """
        Adds robot model to the MJCF model.
        Args:
            mujoco_robot (RobotModel): robot to merge into this MJCF model
        """
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """
        Adds arena model to the MJCF model.
        Args:
            mujoco_arena (Arena): arena to merge into this MJCF model
        """
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def _load_objects_into_model(self, mujoco_objects, object_container, is_visual):
        for obj_name, obj_mjcf in mujoco_objects.items():
            assert(isinstance(obj_mjcf, MujocoGeneratedObject) or isinstance(obj_mjcf, MujocoXMLObject))
            self.merge_asset(obj_mjcf)
            # Load object
            if is_visual:
                obj = obj_mjcf.get_visual(site=False)
            else:
                obj = obj_mjcf.get_collision(site=True)

            for i, joint in enumerate(obj_mjcf.joints):
                obj.append(new_joint(name="{}_jnt{}".format(obj_name, i), **joint))
            object_container.append(obj)
            self.worldbody.append(obj)

    def _set_max_horizontal_radius(self, mujoco_objects_on_table):
        self.max_horizontal_radius = 0
        for _, obj_mjcf in mujoco_objects_on_table.items():
            self.max_horizontal_radius = max(self.max_horizontal_radius, obj_mjcf.get_horizontal_radius())

    def merge_objects(
        self, 
        mujoco_objects_on_table, 
        other_mujoco_objects, 
        visual_mujoco_objects_on_table, 
        other_visual_mujoco_objects
    ):
        """
        Adds object models to the MJCF model.
        Args:
            mujoco_objects (OrderedDict or MujocoObject): objects to merge into this MJCF model
            is_visual (bool): Whether the object is a visual object or not
        """

        self._load_objects_into_model(mujoco_objects_on_table, self.objects_on_table, False)
        self._load_objects_into_model(other_mujoco_objects, self.other_objects, False)
        self._load_objects_into_model(visual_mujoco_objects_on_table, self.visual_objects_on_table, True)
        self._load_objects_into_model(other_visual_mujoco_objects, self.other_visual_objects, True)

        self._set_max_horizontal_radius(mujoco_objects_on_table)


    def place_objects(self):
        """
        Places objects randomly on table until no collisions or max iterations hit.
        """
        pos_arr, _ = self.initializer.sample()
        for i in range(len(self.objects_on_table)):
            self.objects_on_table[i].set("pos", array_to_string(pos_arr[i]))
        return pos_arr