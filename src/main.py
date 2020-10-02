import numpy as np

from mujoco_py import MjSim, MjViewer, MjSimState

import robosuite as suite
from robosuite.environments.base import register_env

from my_environments import Ultrasound


def set_initial_state(sim, old_state, robot):
    old_qvel = old_state.qvel[robot.dof:]
    old_qpos = old_state.qpos[robot.dof:] if robot.gripper.dof < 1 else old_state.qpos[robot.dof-1:]

    init_qvel = [0 for _ in range(robot.dof)]
    initial_model_qvel = np.concatenate((init_qvel, old_qvel), axis=0)
    initial_model_qpos = np.concatenate((robot.init_qpos, old_qpos), axis=0)
        
    new_state = MjSimState(old_state.time, initial_model_qpos, initial_model_qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


register_env(Ultrasound)

env = suite.make(
            'Ultrasound',
            robots='Panda',
            gripper_types='PandaGripper',
            has_renderer=True,            # make sure we can render to the screen
            has_offscreen_renderer=False, # not needed since not using pixel obs
            use_camera_obs=False,         # do not use pixel observations
            control_freq=50,              # control should happen fast enough so that simulation looks smoother
            camera_names='frontview',
        )


# Reset the env
env.reset()

# Get action limits
low, high = env.action_spec

# Run random policy
for t in range(5000):
    env.render()
    action = np.random.uniform(low, high)
    observation, reward, done, info = env.step(action)

# close window
env.close()

'''

# Simulate with mujoco py directly. Does not totally kill my computer hehehh ;) 
world = env.model 
model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
set_initial_state(sim, sim.get_state(), env.robots[0])
viewer = MjViewer(sim)

for _ in range(5000):
    sim.step()
    viewer.render()
'''