"""
Environment that derives from DeformEnv and uses robots instead of anchors
for grasping and control. This class is experimental, so might only be
appropriate for expert users.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import os

import gym
import numpy as np
import pybullet

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray)
from ..utils.bullet_manipulator import BulletManipulator
from ..utils.init_utils import get_preset_properties
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import DEFORM_INFO, ROBOT_INFO

from .deform_env import DeformEnv


class DeformRobotEnv(DeformEnv):
    ORI_SIZE = 3 * 2  # 3D position + sin,cos for 3 Euler angles
    FING_DIST = 0.01  # default finger distance

    def __init__(self, args):
        super(DeformRobotEnv, self).__init__(args)
        act_sz = 3
        if self.food_packing:
            act_sz += DeformRobotEnv.ORI_SIZE
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Wrapped as DeformEnvRobot with act', self.action_space)

    @staticmethod
    def unscale_pos(act, unscaled):
        if unscaled:
            return act
        return act * DeformEnv.WORKSPACE_BOX_SIZE

    def load_objects(self, sim, args, debug):
        res = super(DeformRobotEnv, self).load_objects(sim, args, debug)
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        # print(f"Attempting to load robot, data_path={data_path}")
        sim.setAdditionalSearchPath(data_path)
        # Franka training works ok
        # robot_info = ROBOT_INFO.get(f'franka{self.num_anchors:d}', None)

        # robot_info = ROBOT_INFO.get(f'kinova{self.num_anchors:d}', None)
        robot_info = ROBOT_INFO.get(f'ur5{self.num_anchors:d}', None)
        # print(f"Robot info filename is={robot_info['file_name']}")
        assert(robot_info is not None)  # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])
        if debug:
            print('Loading robot from', robot_path)
        self.robot = BulletManipulator(
            sim, robot_path, control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]),
            global_scaling=robot_info['global_scaling'],
            use_fixed_base=robot_info['use_fixed_base'],
            rest_arm_qpos=robot_info['rest_arm_qpos'],
            left_ee_joint_name=robot_info.get('left_ee_joint_name', None),
            left_ee_link_name=robot_info.get('left_ee_link_name', None),
            left_fing_link_prefix='panda_hand_l_', left_joint_suffix='_l',
            left_rest_arm_qpos=robot_info.get('left_rest_arm_qpos', None),
            debug=debug)
        return res

    def robot_to_anchor(self, object_id, contact_link_id, link_id,
                        object_pos):
        body_pose = pybullet.getLinkState(self.robot.info.robot_id, link_id)
        object_pose = pybullet.getBasePositionAndOrientation(object_id)
        world_to_body = pybullet.invertTransform(
            body_pose[0], body_pose[1])
        object_to_body = pybullet.multiplyTransforms(
            world_to_body[0], world_to_body[1],
            object_pose[0], object_pose[1])
        contact_constraint = pybullet.createConstraint(
            parentBodyUniqueId=self.robot.info.robot_id,
            parentLinkIndex=link_id,
            childBodyUniqueId=object_id,
            childLinkIndex=contact_link_id,
            jointType=self.sim.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0))
            # parentFramePosition=body_pose[0],
            # childFramePosition=object_pos[0])

        return contact_constraint

    #     example self.base_cid = sim.createConstraint(
    #                 self.info.robot_id, -1, -1, -1, sim.JOINT_FIXED, [0.0, 0, 0],
    #                 [0.0, 0, 0], base_pos)

    def make_anchors(self):
        # TODO THE ANCHOR TO THE ROBOT IS DONE HERE, HOW TO FIX IT??
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        assert (preset_dynamic_anchor_vertices is not None)

        for i in range(self.num_anchors):  # make anchors
            # First create the anchor point
            anchor_init_pos = self.args.anchor_init_pos if (i % 2) == 0 else \
                self.args.other_anchor_init_pos
            anchor_id, anchor_pos, anchor_vertices = create_anchor(
                self.sim, anchor_init_pos, i,
                preset_dynamic_anchor_vertices, mesh, radius=0.0001)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)

            # Now anchor the robot to the anchor
            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id

            self.robot_to_anchor(object_id=anchor_id, contact_link_id=anchor_id,
                                 link_id=link_id, object_pos=anchor_pos)

    def do_action(self, action, unscaled=False):
        # Note: action is in [-1,1], so we unscale pos (ori is sin,cos so ok).
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_pos = DeformRobotEnv.unscale_pos(action[0, :3], unscaled)
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:]
        tgt_kwargs = {'ee_pos': tgt_pos, 'ee_ori': tgt_ee_ori,
                      'fing_dist': DeformRobotEnv.FING_DIST}
        if self.num_anchors > 1:  # dual-arm
            res = self.robot.get_ee_pos_ori_vel(left=True)
            left_ee_pos, left_ee_ori = res[0], res[1]
            left_tgt_pos = DeformRobotEnv.unscale_pos(action[1, :3], unscaled)
            left_tgt_ee_ori = left_ee_ori if action.shape[-1] == 3 else \
                action[1, 3:]
            tgt_kwargs.update({'left_ee_pos': left_tgt_pos,
                               'left_ee_ori': left_tgt_ee_ori,
                               'left_fing_dist': DeformRobotEnv.FING_DIST})
        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)
        n_slack = 1  # use > 1 if robot has trouble reaching the pose
        sub_i = 0
        max_diff = 0.02
        diff = self.robot.get_qpos() - tgt_qpos
        while (np.abs(diff) > max_diff).any():
            self.robot.move_to_qpos(
                tgt_qpos, mode=pybullet.POSITION_CONTROL, kp=0.1, kd=1.0)
            self.sim.stepSimulation()
            diff = self.robot.get_qpos() - tgt_qpos
            sub_i += 1
            if sub_i >= n_slack:
                diff = np.zeros_like(diff)  # set while loop to done

    def make_final_steps(self):
        ee_pos, ee_ori, *_ = self.robot.get_ee_pos_ori_vel()
        final_action = np.hstack([ee_pos, ee_ori]).reshape(1, -1)
        if self.num_anchors > 1:  # dual-arm
            left_ee_pos, left_ee_ori, *_ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            final_left_action = np.hstack(
                [left_ee_pos, left_ee_ori]).reshape(1, -1)
            final_action = np.vstack([final_action, final_left_action])
        if self.args.debug:
            print('final_action', final_action)
        info = {'final_obs': []}
        for sim_step in range(DeformEnv.STEPS_AFTER_DONE):
            self.do_action(final_action, unscaled=True)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                next_obs, _ = self.get_obs()
                info['final_obs'].append(next_obs)
        return info

    def get_grip_obs(self):
        grip_obs = []
        ee_pos, _, ee_linvel, _ = self.robot.get_ee_pos_ori_vel()
        grip_obs.extend(ee_pos)
        grip_obs.extend((np.array(ee_linvel) / DeformEnv.MAX_OBS_VEL))
        if self.num_anchors > 1:  # EE pos, vel of left arm
            left_ee_pos, _, left_ee_linvel, _ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            grip_obs.extend(left_ee_pos)
            grip_obs.extend((np.array(left_ee_linvel) / DeformEnv.MAX_OBS_VEL))

        return grip_obs

    def get_reward(self):
        if self.food_packing:
            return self.get_food_packing_reward()
        else:
            return super(DeformRobotEnv, self).get_reward()

    def get_food_packing_reward(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        # rigid_ids[1] is the box, rigid_ids[2] is the can
        box_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[1])
        can_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[2])
        vertex_cent = np.mean(vertex_positions, axis=0)
        dist1 = np.linalg.norm(vertex_cent - box_pos)
        dist2 = np.linalg.norm(vertex_cent - can_pos)

        dist = np.mean([dist1, dist2])
        rwd = -1.0 * dist / DeformEnv.WORKSPACE_BOX_SIZE

        # Squish penalty (to protect the fruit)
        vertices = np.array(vertex_positions)
        relative_dist = np.linalg.norm(vertices - vertices[[0]], axis=1)

        current_shape = relative_dist[self.deform_shape_sample_idx]
        penalty_rwd = np.linalg.norm(current_shape - self.deform_init_shape)
        rwd = rwd + penalty_rwd
        return rwd
