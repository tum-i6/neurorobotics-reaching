
import random

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../grpc/python/communication'))

import communication_client as cc

# for details, please check our correspoding grpc file. This python file is just a wrapper.

class GRPC_rl_connector():
    
    def __init__(self):      
        self.client = cc.GRPCClient()
        ok = self.client.test()
        if (ok!=True):
            print('grpc connect error!')
            sys.exit()
            
    def getCylinderPosition(self):
        cylinderPos = self.client.setup()
        return cylinderPos
    
    def sendActionToBackend(self,action):
        '''
        action = [0] * 6
        for i in range(0, 5):
            action[i] = random.uniform(0, 1)
        '''
        self.action = action
        distanceToTarget = self.client.execute(self.action)
        return distanceToTarget
    
    def getRobotJointStates(self):
        rjs = self.client.get_robot_joint_states()
        return rjs
    
    def getRobotPosition(self):
        rpos = self.client.get_robot_position()
        return rpos
    
    
