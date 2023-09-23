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
from omniisaacgymenvs.robots.articulations.humanoid_custom import Humanoid
# from omniisaacgymenvs.robots.articulations.humanoid import Humanoid

from omniisaacgymenvs.tasks.utils.simple_room import Room
from omniisaacgymenvs.tasks.utils.simple_buffer import SimpleVecBuffer, SimpleExtremeValueChecker

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, quat_from_euler_xyz, get_euler_xyz, quat_rotate, quat_rotate_inverse
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrim, XFormPrimView, GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils import prims

import omni.replicator.core as rep
# from omni.isaac.sensor import Camera

import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2

from pxr import PhysxSchema
from PIL import Image

RECORD = False
# TARGET_POS = [11.5, -6.25, 0]
# TARGET_POS = [-5.0, 6.25, 1.0]

# for Simple room: room_simple_30b15.usd
# TARGET_POS = [-13.0, 0.0, 1.0]
# HUMANOID_POS = [10, 0, 1.34]

# for room A: room_a_50b50.usd
#  -------------.       x 
# |             |   <----.
# |  G          |        | y
# |------o      |        v
# |  S          |   o: origin
# |             |   G: Goal; S: Start
#  -------------'    
# TARGET_POS = [22.0, -13.0, 1.0]   # Goal
# HUMANOID_POS = [13, 13, 1.34]    # Start

POS_DICT = {
    'room_simple_30b15': {
    # .-------------.
    # |             |
    # | S        G  |   o: origin
    # |             |   G: Goal; S: Start
    # '-------------'   
        'target': [-13.0, 0.0, 1.0],
        'humanoid': [10, 0, 1.34],
        'x_range': [-14, 14],
        'y_range': [-7, 7],
    },

    'room_a_50b50': {
    #  -------------.       x 
    # |             |   <----.
    # |  G          |        | y
    # |------o      |        v
    # |  S          |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [22.0, -13.0, 1.0],
        'humanoid': [13, 13, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

    'room_a_50b50_random': {
    #  -------------.       x 
    # |             |   <----.
    # |  G          |        | y
    # |------o      |        v
    # |  S          |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [22.0, -13.0, 1.0],
        'humanoid': [0, 0, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

    'room_a_50b50_simple': {
    #  -------------.       x 
    # |             |   <----.
    # |             |        | y
    # |------o      |        v
    # |  S      G   |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [-13.0, 13.0, 1.0],
        'humanoid': [13, 13, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

    'room_a_50b50_simple2': {
    #  -------------.       x 
    # |        G    |   <----.
    # |             |        | y
    # |------o      |        v
    # |  S          |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [-13.0, -13.0, 1.0],
        'humanoid': [13, 13, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

    'room_a_50b50_debug': {
    #  -------------.       x 
    # |             |   <----.
    # |  G          |        | y
    # |------o      |        v
    # |S            |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [22.0, -13.0, 1.0],
        'humanoid': [23, 2, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

    'room_a_50b50_v2': {
    #  -------------.       x 
    # |             |   <----.
    # |  G          |        | y
    # |------o      |        v
    # |  S          |   o: origin
    # |             |   G: Goal; S: Start
    #  -------------'    
        'target': [22.0, -13.0, 1.0],
        'humanoid': [13, 13, 1.34],
        'x_range': [-25, 25],
        'y_range': [-25, 25],
        'pix_per_meter': 20
    },

}

class BlindHumanoidSimpleHistLocomotionTask(RLTask):
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
        self.forward_weight = self._task_cfg["env"]["forwardWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.height_weight = self._task_cfg["env"]["heightWeight"]

        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]
        self.progress_reward_scale = self._task_cfg["env"]["progressRewardScale"]

        self._room_usd_path = self._task_cfg["env"]['roomUsdPath']
        self._room_name = self._task_cfg["env"]["roomName"]
        self.random_spawn_location = self._task_cfg["env"]["randomSpawnLocation"]
        self.free_coords = None

        self.success_reward = self._task_cfg["env"]["success_reward"]
        self.success_radius = self._task_cfg["env"]["success_radius"]

        self._experiment_id = self._cfg["experiment_id"]
        self._horizon_length = self._cfg['train']['params']['config']['horizon_length']

        # NOTE: for debug only
        self.debug_buf = None

        self._early_observation_clamping = self._task_cfg["env"]["earlyObservationClamping"]

        self.hist_obs_buf = None

        # [NOTE] Experiment Variables
        if self._experiment_id == 0 or self._experiment_id == 1:
            self._num_observations_single_frame = 52
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 2:
            self._num_observations_single_frame = 49
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 3 or self._experiment_id == 4 or self._experiment_id == 5 or self._experiment_id == 6:
            self._num_observations_single_frame = 53    # with up projection
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 7:
            self._num_observations_single_frame = 53 + 21  # with up projection and action
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 8:
            self._num_observations_single_frame = 53 + 21 + 21  # with up projection and action and dof_vel
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 9:
            self._num_observations_single_frame = 53 + 21 + 2  # with up projection and action and gyro
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        elif self._experiment_id == 10:
            self._num_observations_single_frame = 53 + 21 + 21 + 2  # with up projection and action and gyro and dofv
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # check extreme simulation readings
            checker_data_dim = 21*1 #3 + 4 + 6 + 21 + 21 + 4*6
            self.sim_reading_checker = SimpleExtremeValueChecker(self._num_envs, checker_data_dim,self._cfg["sim_device"])


            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?
            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)

        elif self._experiment_id == 11:
            self._num_observations_single_frame = 53 + 21 + 21  # with up projection and action and gyro and dofv, no relative torso position
            self.hist_buffer_size = self._task_cfg['env']['historyLength']

            self._num_observations = self._num_observations_single_frame * self.hist_buffer_size

            # assert self.hist_buffer_size <= self._horizon_length # [TODO]: Simplifies problem for now ?

            self.hist_obs_buf = SimpleVecBuffer(self.hist_buffer_size, self._num_envs, self._num_observations_single_frame, self._cfg["sim_device"], pad='zero')
            self.latest_obs_buf = torch.zeros((self._num_envs, self._num_observations_single_frame), device=self._cfg["sim_device"], dtype=torch.float)
        else:
            raise Exception

        self._num_actions = 21
        # self._humanoid_positions = torch.tensor(HUMANOID_POS)
        self._humanoid_positions = torch.tensor(POS_DICT[self._room_name]['humanoid'])

        self._env = env
        self._global_step = 0

        RLTask.__init__(self, name, env)
        
        return

    def get_free_coords(self, free_map):
        free_coords = np.stack(np.where(free_map == 255)).T 
        free_coords = -1 * free_coords/POS_DICT[self._room_name]['pix_per_meter']
        x_to_center = (POS_DICT[self._room_name]['x_range'][1] - POS_DICT[self._room_name]['x_range'][0])/2
        y_to_center = (POS_DICT[self._room_name]['y_range'][1] - POS_DICT[self._room_name]['y_range'][0])/2
        free_coords[:,[0,1]] = free_coords[:,[1,0]] # swap axis
        free_coords[:,0] = free_coords[:,0] + x_to_center
        free_coords[:,1] = free_coords[:,1] + y_to_center
        return torch.tensor(free_coords, device=self._device)

    def set_up_scene(self, scene) -> None:
        self.get_humanoid()
        self.room = self.get_room()

        # get free map
        free_map = self.room.occ_map
        if free_map is not None:
            free_map = cv2.erode(free_map, np.ones((25, 25), np.uint8), iterations=1) # add 1m buffer
            self.free_coords = self.get_free_coords(free_map)

        # RLTask.set_up_scene(self, scene, replicate_physics=False) # default_ground_plane, GeometryPrim
        RLTask.set_up_scene(self, scene) # default_ground_plane, GeometryPrim NOTE: sim_2022.1.1
        # self._ground_physics = PhysicsMaterial(prim_paths=self._ground_plane_path)

        self._humanoids = ArticulationView(prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False)
        scene.add(self._humanoids)

        self._target = XFormPrimView(prim_paths_expr="/World/envs/.*/Target", reset_xform_properties=False)
        scene.add(self._target)

        # add camera for recording (1 camera only)
        if RECORD:
            record_camera = prims.create_prim(
                prim_path="/World/Camera",
                prim_type="Camera",
                attributes={
                    "focusDistance": 1,
                    "focalLength": 50,
                    "horizontalAperture": 20.955,
                    "verticalAperture": 15.2908,
                    "clippingRange": (0.01, 1000000),
                    "clippingPlanes": np.array([1.0, 0.0, 1.0, 1.0]),
                },
                # translation=(0, 0, 65),
                translation=( 538.0000 + 5,#-1.0120e+03 +5,  
                            1288.0000 - 5,#1.4880e+03 -5, 
                            45),  # env id 3389| envid 2613: -412.0000, 1088.0000
                # envid 2044: 3.8000e+01, 1.4380e+03, 1.3380e+00
                # envid 1401: 538.0000, 1288.0000
                orientation=[0.7071068 , 0, 0, 0.7071068], # wxyz () facing down
            )
            # self.rgb_camera = record_camera
            rep_camera = rep.create.camera(record_camera)   
            render_product = rep.create.render_product(record_camera.GetPrimPath(), resolution=(1024, 1024))
            # render_product = rep.create.render_product(rep_camera, resolution=(512, 512))
            # self.rgb_camera = []
            self.rgb_camera = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_camera.attach([render_product])

            # self.camera = Camera(
            #     prim_path="/World/camera",
            #     position=np.array([0.0, 0.0, 25.0]),
            #     frequency=20,
            #     resolution=(256, 256),
            #     translation=(-1.0120e+03,  1.4880e+03, 65),  # env id 3389
            #     orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
            # )
            # self.camera.initialize()
            # self.camera.add_motion_vectors_to_frame()

            self.rgb_file_path = "/data/zanming/Omniverse/OmniIsaacGymEnvs/omniisaacgymenvs/logs/{}/rgb/".format(self._cfg['experiment'])
            print("RGB_FILE_PATH: ", self.rgb_file_path)
            if not os.path.exists(self.rgb_file_path):
                os.makedirs(self.rgb_file_path, exist_ok=True)

        return


    # Save rgb image to file
    def save_rgb(self, rgb_data, file_name):
        # print('RGB_DATA CLASS', rgb_data.__class__)
        # print()
        rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
        rgba_img = Image.fromarray(rgb_image_data, "RGBA")
        rgb_img = rgba_img.convert('RGB')
        rgb_img.save(file_name + ".jpg")

    def get_camera_data(self): 
        # Generate one frame of data
        # rep.orchestrator.step()
        # Get data
        # rgb = self.rgb_wrist.get_data()
        # depth = self.depth_wrist.get_data()
        print(self._global_step)
        self.save_rgb(self.rgb_camera.get_data(), self.rgb_file_path + "step_%05d"%self._global_step)

        # self.save_rgb(self.camera.get_rgba()[:,:,:], self.rgb_file_path + "step_%05d"%self._global_step)
        return

    def get_humanoid(self):
        humanoid = Humanoid(prim_path=self.default_zero_env_path + "/Humanoid", name="Humanoid", translation=self._humanoid_positions)
        self._sim_config.apply_articulation_settings("Humanoid", get_prim_at_path(humanoid.prim_path), 
            self._sim_config.parse_actor_config("Humanoid"))

    def get_room(self):
        room = Room(prim_path=self.default_zero_env_path +"/Room", name='room', 
                    usd_path=self._room_usd_path, translation=(0, 0, 1.5))
        # room = Room(prim_path=self.default_zero_env_path +"/Room", name='room', translation=(0, 0, 1.5))
        # room = Room(prim_path="/World/Room", name='room', translation=(0, 0, 1.5))

        # create goal
        target_geom = prims.create_prim(
                prim_path=self.default_zero_env_path + "/Target",
                prim_type="Sphere",
                translation=POS_DICT[self._room_name]['target'],
        )
        return room


    def get_robot(self):
        return self._humanoids

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=True)
        velocities = self._robots.get_velocities(clone=True)
        # velocity = velocities[:, 0:3]
        # ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=True)     # [num_env, 21]
        dof_vel = self._robots.get_joint_velocities(clone=True)

        # print('_'*30)
        # print(dof_pos.shape)
        # print(dof_pos[0, :].reshape(-1,3))
        # print(torso_position[0])
        # print(self.target_position[0])
        # print('_'*30)
        # print(self.dof_limits_upper)
        # print(self.dof_limits_lower)

        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces().clone() # (num_envs, num_sensors, 6)

        # clamp observations from simulation by std
        # running mean + std on read data
        # if exceeds threshold?
        # check extreme value, store last value
        # if extreme, load last value, set reset flag
        # assuming extreme value does not appear at first frame (should be valid)

        sim_readings = {'torso_position':torso_position,
            'torso_rotation':torso_rotation,
            'velocities':velocities,
            'dof_pos':dof_pos,
            'dof_vel':dof_vel,
            'sensor_force_torques':sensor_force_torques}
        
        check_readings = {'dof_vel': dof_vel}



        is_invalid = self.sim_reading_checker.check(check_readings)
        if len(is_invalid) > 0:
            last_reading = self.sim_reading_checker.load_last()
            sim_readings['torso_position'][is_invalid] = last_reading['torso_position'][is_invalid]
            sim_readings['torso_rotation'][is_invalid] = last_reading['torso_rotation'][is_invalid]
            sim_readings['velocities'][is_invalid] = last_reading['velocities'][is_invalid]
            sim_readings['dof_pos'][is_invalid] = last_reading['dof_pos'][is_invalid]     # [num_env, 21]
            sim_readings['dof_vel'][is_invalid] = last_reading['dof_vel'][is_invalid]
            sim_readings['sensor_force_torques'][is_invalid] = last_reading['sensor_force_torques'][is_invalid]
             # reset env with extreme value
            self.reset_buf[is_invalid] = 1


        self.sim_reading_checker.update(check_readings)
        self.sim_reading_checker.save_last(sim_readings)

        
        # setting values after filtering
        torso_position = sim_readings['torso_position']
        torso_rotation = sim_readings['torso_rotation']
        velocities = sim_readings['velocities']
        dof_pos = sim_readings['dof_pos']
        dof_vel = sim_readings['dof_vel']
        sensor_force_torques = sim_readings['sensor_force_torques']

        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]



        # [NOTE] Experiment Variables
        if self._experiment_id == 0 or self._experiment_id == 1:
            # [NOTE]: Experiemnt 1_0
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 2:
            # [NOTE]: Observation without goal info
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_ng(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 3 or self._experiment_id == 4 or self._experiment_id == 5 or self._experiment_id == 6:
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 7:
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 8:
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action_dofv(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 9:
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action_normrot(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 10:

            if self._early_observation_clamping:
                self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action_normrot_dofv_clamp(
                    torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                    self.target_position, self.potentials, 
                    self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                    self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                    sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                    self.prev_torso_position
                )
            else:
                self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action_normrot_dofv(
                    torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                    self.target_position, self.potentials, 
                    self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                    self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                    sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                    self.prev_torso_position
                )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        elif self._experiment_id == 11:
            self.latest_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.prev_torso_position[:], self.displacement[:] = get_observations_up_action_normrot_dofv_noPos(
                torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.initial_root_pos, self.initial_root_rot,
                self.target_position, self.potentials, 
                self.dt, self.inv_start_rot, self.basis_vec0, self.basis_vec1, 
                self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self.prev_torso_position
            )

            self.hist_obs_buf.push(self.latest_obs_buf)
            self.obs_buf[:] = self.hist_obs_buf.get()
        else:
            raise Exception

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

        rep.orchestrator.step() # NOTE: added
        if RECORD:
            self.get_camera_data() # save rgb

        self._global_step += 1

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
        if self.random_spawn_location:  # random loc + rot
            root_pos, root_rot = self.get_random_loc_in_range(env_ids, self.initial_root_origin, POS_DICT[self._room_name]['humanoid'], \
                                POS_DICT[self._room_name]['x_range'], POS_DICT[self._room_name]['y_range']), self.get_random_yaw_quaternions(num_resets)
        else:   # random rot only
            root_pos, root_rot = self.initial_root_pos[env_ids],  self.initial_root_rot[env_ids]#self.get_random_yaw_quaternions(num_resets)


        # NOTE: Debug only
        # print(root_pos.shape)
        # root_rot = torch.zeros(num_resets, 4, device=self._device) + torch.tensor([0.85511637, 0. , 0.,  -0.5184361], device=self._device)



        # print('-'*20)
        # print('pos 2613: ', self.initial_root_pos[2613])
        # print('pos 2044: ', self.initial_root_pos[2044])
        # print('pos 1401: ', self.initial_root_pos[1401])
        # update initial root rotation after randomization
        self.initial_root_rot[env_ids] = root_rot
        self.initial_root_pos[env_ids] = root_pos
        # self.init_randomized_root_pos = root_pos


        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        to_target = self.target_position[env_ids] -  self.initial_root_pos[env_ids]
        # to_target = self.target_position[env_ids] -  root_pos
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = torch.norm(to_target, p=2, dim=-1)
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.displacement[env_ids] = torch.zeros((num_resets), device=self._device)
        self.prev_torso_position[env_ids] = root_pos.clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # reset buffer
        if self.hist_obs_buf != None:
            self.hist_obs_buf.reset(env_ids)

        self.sim_reading_checker.reset(env_ids)


        num_resets = len(env_ids)

    def post_reset(self):
        # run once: every time simulation starts

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

        print('-'*20)
        # print(self._humanoids.initialized)
        # print(self._humanoids._dof_names)
        # print(dof_limits)
        # print(self._humanoids._dof_types)
        # print(self._humanoids._dof_indices)

        for m in self._humanoids._dof_names:
            print(m, dof_limits[0, self._humanoids._dof_indices[m]]*180/np.pi)
        print('-'*20)

        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        self._robots = self.get_robot()
        # self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_root_origin, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_root_pos = self.initial_root_origin.clone()

        # print("Initial_root_rot: ", self.initial_root_rot)
        self.initial_dof_pos = self._robots.get_joint_positions()
        # [tzm : nn_4] : initial positial (hands down)
        # self.initial_dof_pos = torch.tensor([0., 0., np.pi/4, -np.pi/4, np.pi/4, -np.pi/4, 0., -np.pi/4, -np.pi/4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device=self._device)
        # self.initial_dof_pos = self.initial_dof_pos[None,:]
        # self.initial_dof_pos = self.initial_dof_pos.repeat(self.num_envs, 1)
        # print("===Shape===: ", self.initial_dof_pos.shape)
        # [END]

        self.target_position, self.target_rotation = self._target.get_world_poses()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # self.targets = torch.tensor(TARGET_POS, dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.displacement = torch.zeros((self.num_envs), device=self._device)
        self.prev_torso_position = self.initial_root_pos.clone()
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        
        latest_obs = self.latest_obs_buf

        # [NOTE] Experiment Variables
        if self._experiment_id == 0:
            # [NOTE]: Experiment 1_0
            self.rew_buf[:] = calculate_metrics(
                latest_obs, self.actions, self.up_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 1 or self._experiment_id == 2:
            # [NOTE]: Experiment 1_1
            self.rew_buf[:] = calculate_metrics_w_progress(
                latest_obs, self.actions, self.up_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 3:
            self.rew_buf[:] = calculate_metrics_w_progress_up(
                latest_obs, self.actions, self.up_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 4:
            self.rew_buf[:] = calculate_metrics_w_progress_up_cont(
                latest_obs, self.actions, self.up_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 5:
            self.rew_buf[:] = calculate_metrics_w_progress_upCont_heightCont(
                latest_obs, self.actions, self.up_weight,self.height_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 6 or self._experiment_id == 7 or self._experiment_id == 8 or self._experiment_id == 9:
            self.rew_buf[:] = calculate_metrics_w_progress_upCont_heightCont_actionCost(
                latest_obs, self.actions, self.up_weight,self.height_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.progress_reward_scale, self.motor_effort_ratio,
                self.displacement
            )
        elif self._experiment_id == 10 or self._experiment_id == 11:
            _, torso_rotation = self._robots.get_world_poses(clone=False)
            velocities = self._robots.get_velocities(clone=False)
            velocity = velocities[:, 0:3]

            # self.rew_buf[:] = calculate_metrics_w_progress_upCont_heightCont_actionCost_fwdCost(
            #     latest_obs, self.actions, self.up_weight,self.height_weight, self.forward_weight, self.potentials, self.prev_potentials,
            #     self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
            #     self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.progress_reward_scale, self.motor_effort_ratio,
            #     self.displacement, velocity, torso_rotation
            # )
            # NOTE: For debug only
            self.rew_buf[:], self.debug_buf = calculate_metrics_w_progress_upCont_heightCont_actionCost_fwdCost_debug(
                latest_obs, self.actions, self.up_weight,self.height_weight, self.forward_weight, self.potentials, self.prev_potentials,
                self.actions_cost_scale, self.energy_cost_scale, self.termination_height, self.success_radius,
                self.death_cost, self.success_reward, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.progress_reward_scale, self.motor_effort_ratio,
                self.displacement, velocity, torso_rotation
            )
        else:
            raise Exception

    def is_done(self) -> None:
        latest_obs = self.latest_obs_buf

        self.reset_buf[:] = is_done(
            latest_obs, self.termination_height, self.success_radius, self.reset_buf, self.progress_buf, self._max_episode_length, self.potentials
        )

    def get_extras(self) -> None:
        # for information logging
        # Initialize: self.extras = {}
        latest_obs = self.latest_obs_buf

        self.extras['is_success'] = is_success(
            latest_obs, self.termination_height, self.success_radius, self.reset_buf, self.progress_buf, self._max_episode_length, self.potentials
        )   # len: num_envs

        self.extras['dist_to_goal'] = self.potentials # len: num_envs
        
        # self.extras['reward_alive'] = self.debug_buf['alive']
        # self.extras['reward_progress'] = self.debug_buf['progress']
        # self.extras['reward_up'] = self.debug_buf['up']
        # self.extras['reward_height'] = self.debug_buf['height']
        # self.extras['reward_actions'] = self.debug_buf['actions']
        self.extras['reward_energy'] = self.debug_buf['energy']
        # self.extras['reward_dof'] = self.debug_buf['dof']
        # self.extras['reward_forward'] = self.debug_buf['forward']
        # self.extras['dones'] = self.reset_buf
        self.extras['init_heading'] = self.initial_root_rot


        # self.extras['torso_position'],  self.extras['torso_rotation'] = self._robots.get_world_poses(clone=False)
        # velocities = self._robots.get_velocities(clone=False)
        # self.extras['torso_velocity'] = velocities[:, 0:3]
        # self.extras['torso_ang_velocity'] = velocities[:, 3:6]
        # self.extras['dof_pos'] = self._robots.get_joint_positions(clone=False)     # [num_env, 21]
        # self.extras['dof_vel'] = self._robots.get_joint_velocities(clone=False)
        
        # self.extras['heading']
        # self.extras['']

        # cleanup
        self.debug_buf = None


    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self.motor_effort_ratio, self.joints_at_limit_cost_scale)

    def get_random_yaw_quaternions(self, num):
        # num, _ = root_quat.shape
        yaw = torch_rand_float(np.pi, -np.pi, (num, 1), device=self._device).squeeze()
        pitch = torch.zeros(num, device=self._device)
        roll = torch.zeros(num, device=self._device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        return quat

    def get_random_loc_in_range(self, env_ids, env_initial_pos, initial_pos, x_range, y_range, z = 1.34, max_thresh=36):
        # num, _ = root_quat.shape

        num = len(env_ids)
        loc = torch.zeros(num, 3,  device=self._device)
        if self.free_coords is not None:
            idx = torch.randint(low=0, high=self.free_coords.shape[0], size=(num,))
            loc[:,:2] = self.free_coords[idx].clone() 
        else:
            loc[:,0] = torch.rand(num, device=self._device) * (x_range[1] - x_range[0]) + x_range[0]
            loc[:,1] = torch.rand(num, device=self._device) * (y_range[1] - y_range[0]) + y_range[0]
        
        loc[:,2] = z


        delta_loc = torch.zeros(num, 3, device=self._device)
        delta_loc[:,0] = loc[:, 0] - initial_pos[0]
        delta_loc[:,1] = loc[:, 1] - initial_pos[1]
        delta_loc[:,2] = loc[:, 2] - initial_pos[2]

        # print('env_initial_pos: ', env_initial_pos[0])

        spawn_loc = env_initial_pos[env_ids] + delta_loc
        if torch.any((torch.sqrt(loc[:,0]**2 + loc[:,1]**2)) > max_thresh):
            print(torch.sqrt(loc[:,0]**2 + loc[:,1]**2))
            raise ValueError("Error: Spawning agent out of bounds")
        return spawn_loc


#####################################################################
###=========================jit functions=========================###
#####################################################################



@torch.jit.script
def get_dof_at_limit_cost(obs_buf, motor_effort_ratio, joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, float) -> Tensor
    # dof_pos_scaled
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 8:29]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 8:29]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
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
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    _, _, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = yaw - torso_init_yaw

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            relative_heading.unsqueeze(-1),                         # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            # actions,
            # dif_vel
        ),
        dim = -1,
    )
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement


@torch.jit.script
def get_observations_ng(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    _, _, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = yaw - torso_init_yaw

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            relative_heading.unsqueeze(-1),                         # shape: 1
            # target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            # actions,
            # dif_vel
        ),
        dim = -1,
    )
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement


@torch.jit.script
def get_observations_up(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    _, _, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = yaw - torso_init_yaw

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            relative_heading.unsqueeze(-1),                         # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            # actions,
            # dif_vel
        ),
        dim = -1,
    )
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement

@torch.jit.script
def get_observations_up_action(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    _, _, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = yaw - torso_init_yaw

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            relative_heading.unsqueeze(-1),                         # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            actions,                                                # shape: num_dof
            # dif_vel
        ),
        dim = -1,
    )
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement

@torch.jit.script
def get_observations_up_action_dofv(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    torso_init_roll, torso_init_pitch, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)

    torso_roll, torso_pitch, torso_yaw = get_euler_xyz(torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = torso_yaw - torso_init_yaw
    relative_pitch = torso_yaw - torso_init_yaw
    relative_roll = torso_pitch - torso_init_pitch
    # print('*****************************************************************************')
    # print('torso_init_rotation: ', torso_init_yaw[0])
    # print('torso_rotation: ', torso_yaw[0])
    # print('yaw: ', yaw[0])
    # print('relative_heading: yaw - init: ', relative_heading[0])

    # torso_rotation == torso_quat, because inv_start_rot always 0

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            normalize_angle(relative_heading).unsqueeze(-1),        # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            actions,                                                # shape: num_dof
            dof_vel * dof_vel_scale,                                # shape: num_dof
        ),
        dim = -1,
    )
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement

@torch.jit.script
def get_observations_up_action_normrot(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    torso_init_roll, torso_init_pitch, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)

    torso_roll, torso_pitch, torso_yaw = get_euler_xyz(torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = torso_yaw - torso_init_yaw
    relative_pitch = torso_pitch - torso_init_pitch
    relative_roll = torso_roll - torso_init_roll
    # print('*****************************************************************************')
    # print('torso_init_rotation: ', torso_init_yaw[0])
    # print('torso_rotation: ', torso_yaw[0])
    # print('yaw: ', yaw[0])
    # print('relative_heading: yaw - init: ', relative_heading[0])

    # torso_rotation == torso_quat, because inv_start_rot always 0

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            normalize_angle(relative_heading).unsqueeze(-1),        # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            actions,                                                # shape: num_dof
            normalize_angle(relative_pitch).unsqueeze(-1),           # shape: 1
            normalize_angle(relative_roll).unsqueeze(-1)            # shape: 1
        ),
        dim = -1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement

@torch.jit.script
def get_observations_up_action_normrot_dofv(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    torso_init_roll, torso_init_pitch, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)

    torso_roll, torso_pitch, torso_yaw = get_euler_xyz(torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = torso_yaw - torso_init_yaw
    relative_pitch = torso_pitch - torso_init_pitch
    relative_roll = torso_roll - torso_init_roll
    # print('*****************************************************************************')
    # print('torso_init_rotation: ', torso_init_yaw[0])
    # print('torso_rotation: ', torso_yaw[0])
    # print('yaw: ', yaw[0])
    # print('relative_heading: yaw - init: ', relative_heading[0])

    # torso_rotation == torso_quat, because inv_start_rot always 0

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame

    # print('**************************************')
    # print(dof_pos_scaled.mean(0))

    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            normalize_angle(relative_heading).unsqueeze(-1),        # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            dof_vel * dof_vel_scale,                                # shape: num_dof
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors(4) * 6    possible [nan]
            actions,                                                # shape: num_dof
            normalize_angle(relative_pitch).unsqueeze(-1),           # shape: 1
            normalize_angle(relative_roll).unsqueeze(-1)            # shape: 1
        ),
        dim = -1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement


@torch.jit.script
def get_observations_up_action_normrot_dofv_clamp(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    torso_init_roll, torso_init_pitch, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)

    torso_roll, torso_pitch, torso_yaw = get_euler_xyz(torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = torso_yaw - torso_init_yaw
    relative_pitch = torso_pitch - torso_init_pitch
    relative_roll = torso_roll - torso_init_roll
    # print('*****************************************************************************')
    # print('torso_init_rotation: ', torso_init_yaw[0])
    # print('torso_rotation: ', torso_yaw[0])
    # print('yaw: ', yaw[0])
    # print('relative_heading: yaw - init: ', relative_heading[0])

    # torso_rotation == torso_quat, because inv_start_rot always 0

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            torso_relative_position[:, :2],                         # shape: 2
            normalize_angle(relative_heading).unsqueeze(-1),        # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            dof_vel * dof_vel_scale,                                # shape: num_dof
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors(4) * 6    possible [nan]
            actions,                                                # shape: num_dof
            normalize_angle(relative_pitch).unsqueeze(-1),           # shape: 1
            normalize_angle(relative_roll).unsqueeze(-1)            # shape: 1
        ),
        dim = -1,
    )

    obs = torch.clamp(obs, min=-10, max=10)
    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement


@torch.jit.script
def get_observations_up_action_normrot_dofv_noPos(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    initial_torso_position,
    initial_torso_rotation,
    target_position,
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
    angular_velocity_scale,
    prev_torso_position
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = target_position - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    # potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    # potentials = torch.norm(velocity, p=2, dim=-1)*dt # [TZM: reward displacement]
    potentials = torch.norm(to_target, p=2, dim=-1) # [TZM: return distance to target to calculate reward]

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, target_position, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper) # scale to [-1, 1]

    displacement = torso_position - prev_torso_position # [in global frame]
    displacement = torch.norm(displacement, p=2, dim=-1)
    # displacement = torch.bmm(displacement.view(num_envs, 1, 3), heading_vec.view(num_envs, 3, 1)).view(num_envs)


    torso_init_roll, torso_init_pitch, torso_init_yaw = get_euler_xyz(initial_torso_rotation) # in global frame (rad)

    torso_roll, torso_pitch, torso_yaw = get_euler_xyz(torso_rotation) # in global frame (rad)


    torso_relative_position = (torso_position - initial_torso_position)
    torso_relative_position = quat_rotate_inverse(initial_torso_rotation, torso_relative_position) # rotate to initial frame

    relative_heading = torso_yaw - torso_init_yaw
    relative_pitch = torso_pitch - torso_init_pitch
    relative_roll = torso_roll - torso_init_roll
    # print('*****************************************************************************')
    # print('torso_init_rotation: ', torso_init_yaw[0])
    # print('torso_rotation: ', torso_yaw[0])
    # print('yaw: ', yaw[0])
    # print('relative_heading: yaw - init: ', relative_heading[0])

    # torso_rotation == torso_quat, because inv_start_rot always 0

    target_relative_position = (target_position - initial_torso_position)
    target_relative_position = quat_rotate_inverse(initial_torso_rotation, target_relative_position) # rotate to initial frame


    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),                       # shape: 1
            up_proj.unsqueeze(-1),                                 # shape: 1
            # torso_relative_position[:, :2],                         # shape: 2
            normalize_angle(relative_pitch).unsqueeze(-1),           # shape: 1
            normalize_angle(relative_roll).unsqueeze(-1),            # shape: 1
            normalize_angle(relative_heading).unsqueeze(-1),        # shape: 1
            target_relative_position[:, :3],                        # shape: 3
            # -------------------------------------------------------------------
            dof_pos_scaled,                                         # shape: num_dof (joint position)
            dof_vel * dof_vel_scale,                                # shape: num_dof
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,   # shape: num_sensors * 6    possible [nan]
            actions,                                                # shape: num_dof
        ),
        dim = -1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec, torso_position, displacement



@torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    success_radius,
    reset_buf,
    progress_buf,
    max_episode_length,
    dist_to_goal
):
    # type: (Tensor, float, float, Tensor, Tensor, float, Tensor) -> Tensor

    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf) # check torso height
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    reset = torch.where(dist_to_goal <= success_radius,  torch.ones_like(reset_buf), reset)
    return reset

@torch.jit.script
def is_success(
    obs_buf,
    termination_height,
    success_radius,
    reset_buf,
    progress_buf,
    max_episode_length,
    dist_to_goal
):
    # type: (Tensor, float, float, Tensor, Tensor, float, Tensor) -> Tensor

    success = torch.where(dist_to_goal <= success_radius,  torch.ones_like(reset_buf), torch.zeros_like(reset_buf))
    return success


@torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale


    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
    )



    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward

@torch.jit.script
def calculate_metrics_w_progress(
    obs_buf,
    actions,
    up_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    # set minimum abs reward
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # print('==========================')
    # print(potentials)
    # print('___________________________')
    # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_bclip
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_up(
    obs_buf,
    actions,
    up_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    # set minimum abs reward
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    up_reward = torch.where(obs_buf[:, 1] > 0.93, up_reward + up_weight, up_reward)

    # print('==========================')
    # print(potentials)
    # print('___________________________')
    # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_bclip
        + up_reward
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_up_cont(
    obs_buf,
    actions,
    up_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    # set minimum abs reward
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    target_up_proj = 1.0    # complete upright
    up_reward = up_weight * torch.abs(obs_buf[:, 1] - target_up_proj)

    # print('==========================')
    # print(potentials)
    # print('___________________________')
    # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_bclip
        + up_reward
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_upCont_heightCont(
    obs_buf,
    actions,
    up_weight,
    height_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float,float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    # set minimum abs reward
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    target_up_proj = 1.0    # complete upright
    up_reward = up_weight * torch.abs(obs_buf[:, 1] - target_up_proj)

    # punish torso height deviate target height # TODO: relative to terrain
    target_height = 1.3
    heigth_reward = height_weight * torch.abs(obs_buf[:, 0] - target_height)

    # print('==========================')
    # print(potentials)
    # print('___________________________')
    # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_bclip
        + up_reward
        + heigth_reward
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_upCont_heightCont_actionCost(
    obs_buf,
    actions,
    up_weight,
    height_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    progress_reward_scale,
    motor_effort_ratio,
    displacement
):
    # type: (Tensor, Tensor, float, float,float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, float, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials  # potential: dist to target, if progress reward < 0 : good progress

    # set minimum abs reward [ at least 1-punish for not going towards goal, at least 1-reward for move towards goal]
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # Penalty for energy used
    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    # action(force) * displaicement * effort_ratio ~= power
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 8+num_dof:8+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)


    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    target_up_proj = 1.0    # complete upright
    up_reward = up_weight * torch.abs(obs_buf[:, 1] - target_up_proj)

    # punish torso height deviate target height # TODO: relative to terrain
    target_height = 1.3
    heigth_reward = height_weight * torch.abs(obs_buf[:, 0] - target_height)

    # print('==========================')
    # # print(potentials)
    # print('alive_reward: ', alive_reward[0])
    # print('progress_reward_bclip: ',progress_reward_bclip[0])
    # print('up_reward: ',up_reward[0])
    # print('heigth_reward: ',heigth_reward[0])
    # print('actions_cost: ',actions_cost[0])
    # print('electricity_cost: ',electricity_cost[0])
    # print('dof_at_limit_cost: ',dof_at_limit_cost[0])
    # # print('___________________________')
    # # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_scale * progress_reward_bclip
        + up_reward
        + heigth_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_upCont_heightCont_actionCost_fwdCost(
    obs_buf,
    actions,
    up_weight,
    height_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    progress_reward_scale,
    motor_effort_ratio,
    displacement,
    root_global_velocity,
    root_global_rotation,
):
    # type: (Tensor, Tensor, float, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, float, Tensor, Tensor, Tensor, Tensor) -> Tensor


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials  # potential: dist to target, if progress reward < 0 : good progress

    # set minimum abs reward [ at least 1-punish for not going towards goal, at least 1-reward for move towards goal]
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # Penalty for energy used
    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    # action(force) * displaicement * effort_ratio ~= power
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 8+num_dof:8+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)


    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    target_up_proj = 1.0    # complete upright
    up_reward = up_weight * torch.abs(obs_buf[:, 1] - target_up_proj)

    # punish torso height deviate target height # TODO: relative to terrain
    target_height = 1.3
    heigth_reward = height_weight * torch.abs(obs_buf[:, 0] - target_height)

    # Foward Cost (fwdCost): encourage velocity align with heading
    # velocity (num_env, 3)
    # root_global_rotation (num_env, 4) quaternion
    _, _, root_global_yaw = get_euler_xyz(root_global_rotation)
    root_vec = torch.stack(
        [torch.cos(root_global_yaw), torch.sin(root_global_yaw)],
        dim = 1)

    max_velocity = 1.5 # max walking velocity m/s
    glb_velocity_norm = torch.norm(root_global_velocity[:,:2], p=2.0, dim=1)[:,None]
    # print('glb_velocity_norm: ', glb_velocity_norm.shape)
    # print('root_global_velocity: ', root_global_velocity.shape)
    # print('root_vec: ', root_vec.shape)
    glb_velocity_vec = (root_global_velocity[:,:2] / glb_velocity_norm)* torch.clamp(glb_velocity_norm, max=max_velocity) # normalize * clamped velocity vec
    # glb_velocity_vec = root_global_velocity[:,:2]
    velocity_proj = torch.sum(root_vec * glb_velocity_vec, dim=1)
    forward_reward = forward_weight * velocity_proj


    # print('==========================')
    # print('yaw: ', root_global_yaw[0])
    # print('root_vec: ', root_vec[0][0], root_vec[0][1])
    # # print(potentials)
    # print('alive_reward: ', alive_reward[0])
    # print('progress_reward_bclip: ',progress_reward_bclip[0])
    # print('up_reward: ',up_reward[0])
    # print('heigth_reward: ',heigth_reward[0])
    # print('actions_cost: ',actions_cost[0])
    # print('electricity_cost: ',electricity_cost[0])
    # print('dof_at_limit_cost: ',dof_at_limit_cost[0])
    # print('forward_reward: ', forward_reward[0])
    # # print('___________________________')
    # # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_scale * progress_reward_bclip
        + up_reward
        + heigth_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
        + forward_reward
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward


@torch.jit.script
def calculate_metrics_w_progress_upCont_heightCont_actionCost_fwdCost_debug(
    obs_buf,
    actions,
    up_weight,
    height_weight,
    forward_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    success_radius,
    death_cost,
    success_reward,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    progress_reward_scale,
    motor_effort_ratio,
    displacement,
    root_global_velocity,
    root_global_rotation,
):
    # type: (Tensor, Tensor, float, float, float, Tensor, Tensor, float, float, float, float, float, float, int, Tensor, float, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Dict[str, Tensor]]


    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials  # potential: dist to target, if progress reward < 0 : good progress

    # set minimum abs reward [ at least 1-punish for not going towards goal, at least 1-reward for move towards goal]
    _ones = torch.ones_like(progress_reward)
    progress_reward_bclip = torch.where(progress_reward > 0, torch.max(_ones, progress_reward), torch.min(-_ones, progress_reward))

    # Penalty for energy used
    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    # action(force) * displaicement * effort_ratio ~= power
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 8+num_dof:8+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)


    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:, 1])
    target_up_proj = 1.0    # complete upright
    up_reward = up_weight * torch.abs(obs_buf[:, 1] - target_up_proj)

    # punish torso height deviate target height # TODO: relative to terrain
    target_height = 1.3
    heigth_reward = height_weight * torch.abs(obs_buf[:, 0] - target_height)

    # Foward Cost (fwdCost): encourage velocity align with heading
    # velocity (num_env, 3)
    # root_global_rotation (num_env, 4) quaternion
    _, _, root_global_yaw = get_euler_xyz(root_global_rotation)
    root_vec = torch.stack(
        [torch.cos(root_global_yaw), torch.sin(root_global_yaw)],
        dim = 1)

    max_velocity = 1.5 # max walking velocity m/s
    glb_velocity_norm = torch.norm(root_global_velocity[:,:2], p=2.0, dim=1)[:,None]
    # print('glb_velocity_norm: ', glb_velocity_norm.shape)
    # print('root_global_velocity: ', root_global_velocity.shape)
    # print('root_vec: ', root_vec.shape)
    glb_velocity_vec = (root_global_velocity[:,:2] / glb_velocity_norm)* torch.clamp(glb_velocity_norm, max=max_velocity) # normalize * clamped velocity vec
    # glb_velocity_vec = root_global_velocity[:,:2]
    velocity_proj = torch.sum(root_vec * glb_velocity_vec, dim=1)
    forward_reward = forward_weight * velocity_proj


    # print('==========================')
    # print('yaw: ', root_global_yaw[0])
    # print('root_vec: ', root_vec[0][0], root_vec[0][1])
    # # print(potentials)
    # print('alive_reward: ', alive_reward[0])
    # print('progress_reward_bclip: ',progress_reward_bclip[0])
    # print('up_reward: ',up_reward[0])
    # print('heigth_reward: ',heigth_reward[0])
    # print('actions_cost: ',actions_cost[0])
    # print('electricity_cost: ',electricity_cost[0])
    # print('dof_at_limit_cost: ',dof_at_limit_cost[0])
    # print('forward_reward: ', forward_reward[0])
    # # print('___________________________')
    # # print(prev_potentials)

    total_reward = (
        torch.zeros_like(alive_reward)
        + alive_reward
        - progress_reward_scale * progress_reward_bclip     # 
        + up_reward
        + heigth_reward                                     #
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost              #
        - dof_at_limit_cost                                 #
        + forward_reward
    )

    # added clipping to rewards for safety
    # total_reward = (
    #     torch.zeros_like(alive_reward)
    #     + alive_reward
    #     - progress_reward_scale *  torch.clip(progress_reward_bclip, -10, 10)
    #     + up_reward
    #     + torch.clip(heigth_reward, -2, 2)
    #     - actions_cost_scale * actions_cost
    #     - energy_cost_scale * torch.clip(electricity_cost, -100, 100)
    #     - torch.clip(electricity_cost, -10, 10)
    #     + forward_reward
    # )
    
    reward_dict = {}
    reward_dict['alive'] = alive_reward
    reward_dict['progress'] = progress_reward_bclip
    reward_dict['up'] = up_reward
    reward_dict['height'] = heigth_reward
    reward_dict['actions'] = actions_cost
    reward_dict['energy'] = electricity_cost
    reward_dict['dof'] = dof_at_limit_cost
    reward_dict['forward'] = velocity_proj

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    #adjust reward for success agents
    total_reward = torch.where(
        potentials <= success_radius, torch.ones_like(total_reward) * success_reward, total_reward
    )
    return total_reward, reward_dict