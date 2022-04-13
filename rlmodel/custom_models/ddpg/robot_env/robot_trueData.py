#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from gym import utils
from robot_env.robot_env_truthData import robotGymEnv



class robottrueDataEnv(robotGymEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.maxtimesteps=150
        robotGymEnv.__init__(
            self, n_substeps=20, distance_threshold=0.10, reward_type=reward_type) # 0.05
        utils.EzPickle.__init__(self)
