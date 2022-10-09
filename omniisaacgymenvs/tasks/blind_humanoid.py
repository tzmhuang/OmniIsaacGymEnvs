# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid
from omniisaacgymenvs.tasks.utils.blind_humanoid_room import Room
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, quat_from_euler_xyz
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

from pxr import PhysxSchema


class BlindHumanoidLocomotionTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]

        self._num_observations = 87
        self._num_actions = 21
        self._humanoid_positions = torch.tensor([3, 3, 1.34])

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_humanoid()
        self.get_room()

        RLTask.set_up_scene(self, scene)
        self._humanoids = ArticulationView(prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False)
        scene.add(self._humanoids)
        return

    def get_humanoid(self):
        humanoid = Humanoid(prim_path=self.default_zero_env_path + "/Humanoid", name="Humanoid", translation=self._humanoid_positions)
        self._sim_config.apply_articulation_settings("Humanoid", get_prim_at_path(humanoid.prim_path), 
            self._sim_config.parse_actor_config("Humanoid"))

    def get_room(self):
        room = Room(prim_path=self.default_zero_env_path +"/Room", name='room', translation=(0, 0, 1.5))
        # room = Room(prim_path="/World/Room", name='room', translation=(0, 0, 1.5))

    def get_robot(self):
        return self._humanoids

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = get_observations(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
            self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale
        )
        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        forces = self.actions * self.joint_gears * self.power_scale

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        # applies joint torques
        self._robots.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
        # dof_pos = torch_rand_float(-0.0, 0.0, (num_resets, self._robots.num_dof), device=self._device) # [tzm : nn_1]
        dof_pos[:] = tensor_clamp(
            self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
        )
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)

        # root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_pos, root_rot = self.initial_root_pos[env_ids], self.get_random_yaw_quaternions(self.initial_root_rot[env_ids])
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self.joint_gears = torch.tensor(
            [
                67.5000, # lower_waist
                67.5000, # lower_waist
                67.5000, # right_upper_arm
                67.5000, # right_upper_arm
                67.5000, # left_upper_arm
                67.5000, # left_upper_arm
                67.5000, # pelvis
                45.0000, # right_lower_arm
                45.0000, # left_lower_arm
                45.0000, # right_thigh: x
                135.0000, # right_thigh: y
                45.0000, # right_thigh: z
                45.0000, # left_thigh: x
                135.0000, # left_thigh: y
                45.0000, # left_thigh: z
                90.0000, # right_knee
                90.0000, # left_knee
                22.5, # right_foot
                22.5, # right_foot
                22.5, # left_foot
                22.5, # left_foot
            ],
            device=self._device,
        )

        self.max_motor_effort = torch.max(self.joint_gears)
        self.motor_effort_ratio = self.joint_gears / self.max_motor_effort
        dof_limits = self._humanoids.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        print("Initial_root_rot: ", self.initial_root_rot)
        self.initial_dof_pos = self._robots.get_joint_positions()
        # [tzm : nn_4] : initial positial (hands down)
        self.initial_dof_pos = torch.tensor([0., 0., np.pi/4, -np.pi/4, np.pi/4, -np.pi/4, 0., -np.pi/4, -np.pi/4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device=self._device)
        self.initial_dof_pos = self.initial_dof_pos[None,:]
        self.initial_dof_pos = self.initial_dof_pos.repeat(self.num_envs, 1)
        print("===Shape===: ", self.initial_dof_pos.shape)
        # [END]

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([11.5, -6.25, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = calculate_metrics(
            self.obs_buf, self.actions, self.up_weight, self.heading_weight, self.potentials, self.prev_potentials,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio
        )

    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length, self.potentials
        )

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self.motor_effort_ratio, self.joints_at_limit_cost_scale)

    def get_random_yaw_quaternions(self, root_quat):
        num, _ = root_quat.shape
        yaw = torch_rand_float(np.pi, -np.pi, (num, 1), device=self._device).squeeze()
        pitch = torch.zeros(num, device=self._device)
        roll = torch.zeros(num, device=self._device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        return quat

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, motor_effort_ratio, joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, float) -> Tensor
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
    )
    return dof_at_limit_cost

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(angle_to_target).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            heading_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec

@torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length,
    dist_to_goal
):
    # type: (Tensor, float, Tensor, Tensor, float, Tensor) -> Tensor
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    reset = torch.where(dist_to_goal <= 0.25,  torch.ones_like(reset_buf), reset)
    return reset

@torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(
        obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8
    )

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    # action(force) * velocity * effort_ratio ~= power
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    # progress_reward = potentials - prev_potentials
    progress_reward = potentials # [TZM: reward displacement]
    disp_reward = obs_buf[:,1] #[TZM: X axis of local velocity]
    # print("Rewards: ")
    # print('--------------------------------')
    # print(progress_reward)
    # print('================================')
    # print(disp_reward)
    # print('+++++++++++++++++++++++++++++++++')
    # print(heading_reward)
    total_reward = (
        progress_reward * 0.0
        + alive_reward
        + up_reward
        + disp_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )
    total_reward = torch.where(
        potentials <= 0.25, torch.ones_like(total_reward) * 5000, total_reward
    )
    return total_reward