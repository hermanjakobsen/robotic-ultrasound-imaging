import numpy as np

from mujoco_py import MjSim, MjViewer, MjSimState

import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.models.grippers import GRIPPER_MAPPING

from my_environments import Ultrasound
from my_models.grippers import UltrasoundProbeGripper


def set_initial_state(sim, old_state, robot):
    ''' Used for setting initial state when simulating with mujoco-py directly '''

    old_qvel = old_state.qvel[robot.dof:]
    old_qpos = old_state.qpos[robot.dof:] if robot.gripper.dof < 1 else old_state.qpos[robot.dof-1:]

    init_qvel = [0 for _ in range(robot.dof)]
    initial_model_qvel = np.concatenate((init_qvel, old_qvel), axis=0)
    initial_model_qpos = np.concatenate((robot.init_qpos, old_qpos), axis=0)
        
    new_state = MjSimState(old_state.time, initial_model_qpos, initial_model_qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


def register_gripper(gripper_class):
    GRIPPER_MAPPING[gripper_class.__name__] = gripper_class


def relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd):
    assert len(goal_joint_pos) == robot.dof

    action = [0 for _ in range(robot.dof)]
    curr_joint_pos = robot._joint_positions
    curr_joint_vel = robot._joint_velocities

    for i in range(robot.dof):
        action[i] = (goal_joint_pos[i] - curr_joint_pos[i]) * kp - curr_joint_vel[i] * kd
    
    return action


def robosuite_simulation_controller_test(env, sim_time):
    # Reset the env
    env.reset()

    robot = env.robots[0]
    goal_joint_pos = [np.pi / 2, 0, 0, 0, 0, 0]
    kp = 2
    kd = 1.2
    # Run random policy
    for t in range(sim_time):
        if env.done:
            break
        env.render()
        
        action = relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd)

        if t > 1200:
            action = relative2absolute_joint_pos_commands([0, -np.pi/4, 0, 0, 0, 0], robot, kp, kd)
        elif t > 800:
            action = relative2absolute_joint_pos_commands([0, 0, 0, 0, 0, 0], robot, kp, kd)
        elif t > 400:
            action = relative2absolute_joint_pos_commands([np.pi, 0, 0, 0, 0, 0], robot, kp, kd)

        observation, reward, done, info = env.step(action)

    # close window
    env.close() 


def mujoco_py_simulation(env, sim_time):
    world = env.model 
    model = world.get_model(mode="mujoco_py")

    sim = MjSim(model)
    set_initial_state(sim, sim.get_state(), env.robots[0])
    viewer = MjViewer(sim)

    for _ in range(sim_time):
        sim.step()
        viewer.render()
 
register_env(Ultrasound)
register_gripper(UltrasoundProbeGripper)

env = suite.make(
            'Ultrasound',
            robots='UR5e',
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,
            horizon=2000      
        )


#robosuite_simulation_controller_test(env, env.horizon)
#mujoco_py_simulation(env, env.horizon)


############### CODE FOR TESTING THE TUNING OF SOFT TORSO PARAMETERS ###############
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import EmptyArena
from robosuite.utils.mjcf_utils import new_joint

from my_models.objects import SoftTorsoObject, BoxObject

world = MujocoWorldBase()
arena = EmptyArena()
arena.set_origin([0, 0, 0])
world.merge(arena)

soft_torso = SoftTorsoObject()
obj = soft_torso.get_collision()

box = BoxObject()
box_obj = box.get_collision()

obj.append(new_joint(name='soft_torso_free_joint', type='free'))
box_obj.append(new_joint(name='box_free_joint', type='free'))

world.merge_asset(soft_torso)

world.worldbody.append(obj)
world.worldbody.append(box_obj)
model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)


for i in range(10000):
  sim.step()
  viewer.render()