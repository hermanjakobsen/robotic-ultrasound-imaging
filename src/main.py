import robosuite as suite
import numpy as np
from mujoco_py import MjSim, MjViewer, MjSimState

from robosuite.environments.base import register_env
from my_environments import PandaUltrasound, UR5Ultrasound


def set_initial_state(sim, old_state, robot):

    initial_model_qpos = np.concatenate((robot.init_qpos, old_state.qpos[robot.dof:]), axis=0)
    initial_model_qvel = np.concatenate((robot.init_qvel, old_state.qvel[robot.dof:]), axis=0)

    new_state = MjSimState(old_state.time, initial_model_qpos, initial_model_qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


register_env(UR5Ultrasound)
register_env(PandaUltrasound)

#env = suite.make("PandaUltrasound", has_renderer=True)
env = suite.make("UR5Ultrasound", has_renderer=True)

'''
env.reset()
for _ in range(5000):
    action = [0 for _ in range(env.dof)]
    env.step(action)
    env.render()
'''


# Simulate with mujoco py directly. Does not totally kill my computer hehehh ;) 
world = env.model 
model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
set_initial_state(sim, sim.get_state(), world.robot)
viewer = MjViewer(sim)

for _ in range(5000):
    sim.step()
    viewer.render()
