from robosuite.models.tasks import Task, UniformRandomSampler
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class UltrasoundTask(Task):
    """
    Creates MJCF model of an ultrasound task.
    A task consists of one robot holding an ultrasound probe and interacting with 
    a soft body in various settings. This class combines the robot, the table
    arena, and the objects into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, mujoco_objects_on_table, initializer=None):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects, mujoco_objects_on_table)

        if initializer is None:
            initializer = UniformRandomSampler(z_rotation=None)
        mjcfs = [x for _, x in mujoco_objects_on_table.items()]

        self.initializer = initializer
        self.initializer.setup(mjcfs, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects, mujoco_objects_on_table):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects_on_table= mujoco_objects_on_table

        self.objects = []  # xml manifestation
        self.objects_on_table = []  # xml manifestation

        self._load_objects_into_model(mujoco_objects, self.objects)
        self._load_objects_into_model(mujoco_objects_on_table, self.objects_on_table)


    def _load_objects_into_model(self, mujoco_objects, object_container):
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            object_container.append(obj)
            self.worldbody.append(obj)

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pos_arr, _ = self.initializer.sample()

        for i in range(len(self.objects_on_table)):
            self.objects_on_table[i].set("pos", array_to_string(pos_arr[i]))
