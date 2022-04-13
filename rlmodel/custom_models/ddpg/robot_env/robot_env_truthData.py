#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import os
import math
from gym.utils import seeding
import random
import  time
from gym import spaces
from arguments import get_args, Args
import sys

# import the grpc path such that this file can communicate with nrp via grpc

sys.path.insert(1, os.path.join(sys.path[0], '../../grpc/python/communication'))
import experiment_api_wrapper as eaw


'''
Modified by Chuanlong Zang
HER + action clip scale
start from 2021 4.16
'''

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class robotGymEnv(gym.Env):
    """Superclass for all robot environments.
    """
    def __init__(
        self, n_substeps, distance_threshold, reward_type,
    ):
        # Initializes a new reach environment.

        """Args:
           distance_threshold (float)
           reward_type ('sparse' or 'dense')
        """
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.seed()

        # The robot arm has 6 joints to execute.
        action_dim = 6

        # max und min joint angles
        self._action_bound = math.pi

        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high) 
        self.experiment = eaw.ExperimentWrapper()
        self.reset()

        # GoalEnv methods

    # ----------------------------
    
    def reset(self):
        # get the position of cylinder (target Position)

        self.experiment.cylinder.random_reset()
        
        cylinderPosition = np.array(self.experiment.cylinder.get_position())

        print("cylinderPosition is: ", cylinderPosition)
        
        self.goal = cylinderPosition
        obs = self._get_obs()
        self._observation = obs
        return self._observation
 
    def _get_obs(self):

        # Observations are going to be ground truth data 
        # from the simulation such as the target pose and the current joint angles 
        
        robotJointState = np.array(self.experiment.robot.get_joint_states())
        robotPosition = np.array(self.experiment.robot.get_position())
        eePosition = robotPosition[3]
        relativePosition = self.goal - eePosition
        cylinderPosition = self.goal

        obs = np.concatenate((
            robotJointState.flatten(),    # 6
            eePosition.flatten(),         # 3
            cylinderPosition.flatten(),   # 3
            relativePosition.flatten()))  # 3

        # here adapts the sb3 settings

        achieved_goal = eePosition.copy()
        target_pos = self.goal.copy()
        
        self._observation = obs
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pos, # .flatten()
        }

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def step(self, action):
        action = np.clip(action,-math.pi,math.pi) # dim 6
        
        self._set_action(action)

        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        
        return obs, reward, done, info


    def _set_action(self, action):
        
        oldRobotJointState = np.array(self.experiment.robot.get_joint_states())
        setState = np.clip(oldRobotJointState + action,-math.pi,math.pi)
        distanceToTarget = self.experiment.robot.act(setState)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        print(d)
        return (d < self.distance_threshold).astype(np.float32)
