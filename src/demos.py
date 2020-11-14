import numpy as np
import glfw

from mujoco_py import MjSim, MjViewer

import robosuite as suite
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import EmptyArena
from robosuite.utils.mjcf_utils import new_joint

from my_models.objects import SoftTorsoObject, BoxObject
from helper import relative2absolute_joint_pos_commands, set_initial_robot_state, transform_ee_frame_axes


def robosuite_simulation_controller_test(env, experiment_name, save_data=False):
    # Reset the env
    env.reset()

    sim_time = env.horizon

    robot = env.robots[0]
    joint_pos_obs = np.empty(shape=(sim_time, robot.dof))
    ref_values = np.array([np.pi/2, 3*np.pi/2, -np.pi/4])

    time_scaler = 3 if robot.controller_config['type'] == 'JOINT_POSITION' else 1

    goal_joint_pos = [ref_values[0], 0, 0, 0, 0, 0]
    kp = 2
    kd = 1.2

    for t in range(sim_time):
        if env.done:
            break
        env.render()
        
        action = relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd)

        if t > 1200*time_scaler:
            action = relative2absolute_joint_pos_commands([0, ref_values[2], 0, 0, 0, 0], robot, kp, kd)
        elif t > 800*time_scaler:
            action = relative2absolute_joint_pos_commands([0, 0, 0, 0, 0, 0], robot, kp, kd)
        elif t > 400*time_scaler:
            action = relative2absolute_joint_pos_commands([ref_values[1], 0, 0, 0, 0, 0], robot, kp, kd)

        observation, reward, done, info = env.step(action)
        joint_pos_obs[t] = observation['robot0_joint_pos']

    # close window
    env.close() 

    if save_data:
        np.savetxt('data/'+experiment_name+'_joint_pos_controller_test.csv', joint_pos_obs, delimiter=",")
        np.savetxt('data/'+experiment_name+'_ref_values_controller_test.csv', ref_values, delimiter=",")


def robosuite_simulation_contact_btw_probe_and_body(experiment_name, save_data=False):
    
    for episode in range(3):
        env = suite.make(
            'Ultrasound',
            robots='UR5e',
            controller_configs=None,
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,
            horizon=200      
        )
    
        # Reset the env
        env.reset() 

        sim_time = env.horizon
        robot = env.robots[0]
        num_robot_joints = robot.dof - 1 if robot.gripper.dof > 0 else robot.dof

        joint_torques = np.empty(shape=(sim_time, num_robot_joints))
        ee_forces = np.empty(shape=(sim_time, 3))
        ee_torques = np.empty(shape=(sim_time, 3))
        contact = np.empty(shape=(sim_time, 1))
        
        time_scaler = 3 if robot.controller_config['type'] == 'JOINT_POSITION' else 1

        goal_joint_pos = [0, -np.pi/4, np.pi/3, -np.pi/2, -np.pi/2, 0]
        goal_joint_pos = goal_joint_pos + [0] if robot.gripper.dof > 0 else goal_joint_pos
        kp = 2
        kd = 1.2

        for t in range(sim_time):
            if env.done:
                break
            env.render()
            
            action = relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd)

            if t > 400*time_scaler:
                goal_joint_pos = [0, -np.pi/4, np.pi, -np.pi/2, -np.pi/2, 0]
                goal_joint_pos = goal_joint_pos + [0] if robot.gripper.dof > 0 else goal_joint_pos

                action = relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd)

            observation, reward, done, info = env.step(action)

            joint_torques[t] = robot.torques 
            ee_forces[t] = transform_ee_frame_axes(robot.ee_force) if robot.gripper_type == 'UltrasoundProbeGripper' else robot.ee_force
            ee_torques[t] = transform_ee_frame_axes(robot.ee_torque) if robot.gripper_type == 'UltrasoundProbeGripper' else robot.ee_torque
            if robot.has_gripper:
                contact[t] = observation['contact'][0]

        if save_data:
            np.savetxt('data/'+experiment_name+str(episode)+'_joint_torques_contact_btw_probe_and_body.csv', joint_torques, delimiter=",")
            np.savetxt('data/'+experiment_name+str(episode)+'_ee_forces_contact_btw_probe_and_body.csv', ee_forces, delimiter=",")
            np.savetxt('data/'+experiment_name+str(episode)+'_ee_torques_contact_btw_probe_and_body.csv', ee_torques, delimiter=",")
            np.savetxt('data/'+experiment_name+str(episode)+'_contact_contact_btw_probe_and_body.csv', contact, delimiter=",")

        # close window
        env.close() 


def mujoco_py_simulation(env):
    sim, viewer = create_sim_and_viewer(env)

    for _ in range(env.horizon):
        sim.step()
        viewer.render()
        

def create_sim_and_viewer(env):
    world = env.model 

    soft_torso = world.other_mujoco_objects['soft_torso']
    composite = soft_torso._get_composite_element()
    #print(composite.get('solrefsmooth'))
   # print(world.get_xml())

    model = world.get_model(mode="mujoco_py")
    sim = MjSim(model)
    set_initial_robot_state(sim, env.robots[0])
    viewer = MjViewer(sim)

    return sim, viewer


def change_parameters_of_soft_body_demo(episodes):
 
    for _ in range(episodes):

        env = suite.make(
            'Ultrasound',
            robots='UR5e',
            controller_configs=None,
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,
            horizon=800      
        )

        sim, viewer = create_sim_and_viewer(env)
    
        for _ in range(env.horizon):
            sim.step()
            viewer.render()
        
        glfw.destroy_window(viewer.window)


def body_softness_test():
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

    for _ in range(10000):
   
        sim.step()
        viewer.render()