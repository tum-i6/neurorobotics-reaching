import numpy as np
from time import sleep 

import gym  
from gym import Env, GoalEnv, spaces
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from spawn_cylinder import augment_cylinder_position
from train_helpers import get_action_space, threshold_scheduling, get_reward
from utils import scatter_plotter, bound_space, bound_space_her, normalize_space, de_normalize_space

# HER required GoalEnv
class SimGoalEnv(gym.GoalEnv):
    """ Custom Goal-based Environment class having the same method structure as the environments in gym.
        Can be used as environment for models with HER (background: HER needs a goal-based environment).
       
        Args:
            experiment_wrapper: client to communicate via gRPC with the simulation backend (to call experiment_api)  
            params: stack of parameters
            writer: instance of SummaryWriter (tensorboard logging)
    """
    def __init__(self, exp, params, writer, cnn):
        super(SimGoalEnv, self).__init__()
    
        # experiment_api wrapper
        self.sim = exp
        
        # ensure that all keys are in the param list
        if "img_pi_data" not in params:
            params["img_pi_data"] = 0
        if "SPACE_NORM" not in params:
            params["SPACE_NORM"] = 0

        # CNN model for image recognition
        self.cnn = cnn
            
        # initial threshold for done
        self.threshold = params["THRESHOLD"]
        
        if params["img_pi_data"] == 1:                                     
            self.train_loader = params["train_loader"] 
            self.train_dataiter = iter(self.train_loader)
            self.test_loader = params["test_loader"] 
            self.test_dataiter = iter(self.test_loader)
        
        # Define action and observation space 
        # observation-space (ee-pos, cyl-pos, joints) 
        # -> HER requires division into 'desired_goal', 'achieved_goal' and 'observation'
        self.space_desired_goal_low = np.float32(params['OBJ_SPACE_LOW'][3:6])
        self.space_desired_goal_high = np.float32(params['OBJ_SPACE_HIGH'][3:6])

        self.space_achieved_goal_low = np.float32(params['OBJ_SPACE_LOW'][:3])
        self.space_achieved_goal_high = np.float32(params['OBJ_SPACE_HIGH'][:3])

        self.space_obj_low = np.float32(params['OBJ_SPACE_LOW'])
        self.space_obj_high = np.float32(params['OBJ_SPACE_HIGH'])

        if params["SPACE_NORM"] == 1:
            # set the observation space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
            self.observation_space = bound_space_her(self.space_obj_low , self.space_obj_high, self.space_desired_goal_low, self.space_desired_goal_high, self.space_achieved_goal_low, self.space_achieved_goal_high)
            # retrive the real values for the selected action space
            self.orig_act_space = get_action_space(params)
            # set the action space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
            self.action_space = bound_space(self.orig_act_space.low, self.orig_act_space.high)
        else:
            self.observation_space = spaces.Dict(dict(
                                                    desired_goal = spaces.Box(low=self.space_desired_goal_low, high=self.space_desired_goal_high),
                                                    achieved_goal = spaces.Box(low=self.space_achieved_goal_low, high=self.space_achieved_goal_high),
                                                    observation = spaces.Box(low=self.space_obj_low, high=self.space_obj_high)
                                                ))               
                                  
            # action-space (joints 1-6) 
            self.action_space = get_action_space(params)

        # other vars
        self.pos_cyl = []
        if params["img_pi_data"] == 1:
            self.pos_cyl_gt = [0] * 3
        self.r_js = []
        self.dist = 0
        
        # episode
        self.episode_length = 0
        
        # history (for threshold scheduling)
        self.history = []
        
        # data (for plotting and logging)
        self.data = {}
        self.data["x"] = []
        self.data["y"] = []
        self.data["c"] = []
        self.rewards=[]
        self.distances=[]
        
        # eval
        self.eval = False
    
        # stepper for tensorboard writer
        self.stepper = params["GLOBAL_STEPPER"]

        # params serving as arguments in function calls
        self.params = params

        #Tensorboard SummaryWriter
        self.writer = writer
        
    def step(self, action):
        print("  step")

        if self.params["SPACE_NORM"] == 1:
            action = de_normalize_space(action, self.orig_act_space.low, self.orig_act_space.high, self.action_space.low, self.action_space.high)

        # execute action
        rcd = self.sim.execute(action)
                    
        # make observation
        self.r_js = self.sim.robot.get_joint_states()
        _, _, _, pos_ee = self.sim.robot.get_position()
        
        # create observation
        obs = {}
        obs['observation'] = np.float32((pos_ee[0], pos_ee[1], pos_ee[2], self.pos_cyl_cnn[0], self.pos_cyl_cnn[1], self.pos_cyl[2], self.r_js[0], self.r_js[1], self.r_js[2], self.r_js[3], self.r_js[4], self.r_js[5]))        
        obs['desired_goal'] = np.float32((self.pos_cyl_cnn[0], self.pos_cyl_cnn[1], self.pos_cyl[2]))
        obs['achieved_goal'] = np.float32((pos_ee[0], pos_ee[1], pos_ee[2]))
        
        # normalize the observation space between -1 and 1 
        if self.params["SPACE_NORM"] == 1:
            obs['observation'] = normalize_space(obs['observation'], self.space_obj_low, self.space_obj_high, self.observation_space['observation'].low, self.observation_space['observation'].high)  
            obs['desired_goal'] = normalize_space(obs['desired_goal'], self.space_desired_goal_low, self.space_desired_goal_high, self.observation_space['desired_goal'].low, self.observation_space['desired_goal'].high)  
            obs['achieved_goal'] = normalize_space(obs['achieved_goal'], self.space_achieved_goal_low, self.space_achieved_goal_high, self.observation_space['achieved_goal'].low, self.observation_space['achieved_goal'].high)  

        # distance
        if self.params["img_pi_data"] == 1:
            self.dist = np.linalg.norm(np.asarray(self.pos_cyl_gt, dtype=np.float32) - np.asarray(pos_ee, dtype=np.float32))
        else:
            self.dist = np.linalg.norm(np.asarray(self.pos_cyl, dtype=np.float32) - np.asarray(pos_ee, dtype=np.float32))
        self.distances.append(self.dist)
               
        # reward
        reward = get_reward(self.dist, self.threshold, self.params)
        self.rewards.append(reward)
        
        # done
        if reward >= 1.0:
            done = True
            self.history.append(1)
            self.data["c"].append("green")
        else:
            done = False
            self.history.append(0)
            self.data["c"].append("red")

            
        # increment episode_length counter
        self.episode_length += 1
        
        # reset env if max_episode_length is reached
        if self.params["MAX_EPISODE_LENGTH"]:
            if self.episode_length >= self.params["MAX_EPISODE_LENGTH"]:
                done = True
                print("  [step]: max_episode_length reached")
                    
        # do the following only in training mode
        if not self.eval:
            # threshold scheduling
            if self.params["THRESHOLD_SCHEDULING"]:
                self.threshold, self.history = threshold_scheduling(self.history, self.threshold, self.params)

            # tensorboard writer
            self.writer.add_scalar("distance", self.dist, self.stepper)
            self.writer.add_scalar("reward", reward, self.stepper)
            self.writer.add_scalar("threshold", self.threshold, self.stepper)
            self.writer.add_scalar("avg_reward", np.mean(self.rewards[-100:]), self.stepper)
            self.writer.add_scalar("avg_distance", np.mean(self.distances[-100:]), self.stepper)
            self.stepper += 1

            # plot data
            scatter_plotter(self.data, self.stepper)
            
        # info        
        info = {}
        info['total_distance'] = self.dist
        info['cyl position'] = self.pos_cyl
        info['ee position'] = pos_ee
        info['joint position'] = self.r_js
        
        return obs, reward, done, info
        
     
    def reset(self):
        print("new episode")
        # reset episode_length counter
        self.episode_length = 0

        # make observation
        if self.params["img_pi_data"] == 1:
            if self.eval:
                retrived_data = self.test_dataiter.next()
            else: 
                retrived_data = self.train_dataiter.next()
            self.pos_cyl = retrived_data[0][0]
            self.pos_cyl_gt = retrived_data[1][0]
            print("Detection: ", self.pos_cyl, " | GroundT  : ", self.pos_cyl_gt)
        else:
            self.pos_cyl = self.sim.setup()

            # wait for the robot to be in its initial position for the image
            sleep(2)

            # get image from simulation
            self.image = self.sim.camera.get_image()

            # retrieve the cylinder position from image with CNN
            self.image = (F.normalize(torch.tensor(self.image).type(torch.float), dim=2) - 0.5) * 2
            self.pos_cyl_cnn = (self.cnn(self.image.reshape(1, -1, 120, 120)).squeeze().detach().numpy()) / 10

        self.r_js = self.sim.robot.get_joint_states()  
        _, _, _, pos_ee = self.sim.robot.get_position()

        # augment cylinder pos
        # self.pos_cyl = augment_cylinder_position(self.pos_cyl, self.params)
    
        # data 
        self.data["x"].append(self.pos_cyl[0])
        self.data["y"].append(self.pos_cyl[1])
        
        # create observation
        obs = {}
        obs['observation'] = np.array((pos_ee[0], pos_ee[1], pos_ee[2], self.pos_cyl_cnn[0], self.pos_cyl_cnn[1], self.pos_cyl[2], self.r_js[0], self.r_js[1], self.r_js[2], self.r_js[3], self.r_js[4], self.r_js[5]), dtype=np.float32)                
        obs['desired_goal'] = np.array((self.pos_cyl_cnn[0], self.pos_cyl_cnn[1], self.pos_cyl[2]))
        obs['achieved_goal'] = np.array((pos_ee[0], pos_ee[1], pos_ee[2]))
        
        if self.params["SPACE_NORM"] == 1:
            obs['observation'] = normalize_space(obs['observation'], self.space_obj_low, self.space_obj_high, self.observation_space['observation'].low, self.observation_space['observation'].high)  
            obs['desired_goal'] = normalize_space(obs['desired_goal'], self.space_desired_goal_low, self.space_desired_goal_high, self.observation_space['desired_goal'].low, self.observation_space['desired_goal'].high)  
            obs['achieved_goal'] = normalize_space(obs['achieved_goal'], self.space_achieved_goal_low, self.space_achieved_goal_high, self.observation_space['achieved_goal'].low, self.observation_space['achieved_goal'].high)  

        return obs
    

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Compute the step reward. This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved. If you wish to include
            additional rewards that are independent of the goal, you can include the necessary values to derive it in info and compute it accordingly.
        
            Args:
                achieved_goal (object): the goal that was achieved during execution
                desired_goal (object): the desired goal that we asked the agent to attempt to achieve
                info (dict): an info dictionary with additional information
            
            Returns:
                reward (float): The reward that corresponds to the provided achieved goal w.r.t. to the desired goal. 
                                Note that the following should always hold true:
                                    ob, reward, done, info = env.step()
                                    assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        # distance
        self.dist = np.linalg.norm(np.asarray(achieved_goal, dtype=np.float32) - np.asarray(desired_goal, dtype=np.float32))
        
        # threshold scheduling
        if self.params["THRESHOLD_SCHEDULING"]:
            self.threshold, self.history = threshold_scheduling(self.history, self.threshold, self.params)
          
        # reward
        reward = get_reward(self.dist, self.threshold, self.params)
        
        return reward
    
    def get_threshold(self):
        return self.threshold
    
    def set_eval(self, ev=True):
        self.eval = ev
