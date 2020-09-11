from robosuite.models import MujocoWorldBase
from my_objects.xml_objects import BreadObject, TorsoObject

from mujoco_py import MjSim, MjViewer

world = MujocoWorldBase()

object_mjcf = TorsoObject()
world.merge_asset(object_mjcf)

obj = object_mjcf.get_collision(name='torso', site=True)
world.worldbody.append(obj)

model = world.get_model(mode="mujoco_py")
sim = MjSim(model)
viewer = MjViewer(sim)

for i in range(10000):

    sim.step()
    viewer.render()

