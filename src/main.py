import robosuite as suite
import numpy as np
from mujoco_py import MjSim, MjViewer

from robosuite.environments.base import register_env
from my_environments import PandaUltrasound, UR5Ultrasound

register_env(UR5Ultrasound)
register_env(PandaUltrasound)
env = suite.make("PandaUltrasound", has_renderer=True)
#env = suite.make("UR5Ultrasound", has_renderer=True)
world = env.model 

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)

for _ in range(5000):
    sim.step()
    viewer.render()