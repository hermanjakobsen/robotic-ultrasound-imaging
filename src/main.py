import robosuite as suite
import numpy as np
from mujoco_py import MjSim, MjViewer

from robosuite.environments.base import register_env
from my_environments import SawyerUltrasound

register_env(SawyerUltrasound)
env = suite.make("SawyerUltrasound", has_renderer=True)
world = env.model 

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)

for _ in range(5000):
    sim.step()
    viewer.render()