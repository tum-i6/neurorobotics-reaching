# Author: Kevin Farkas (5.Mai 2021)
#
# content:
# - simple communication-client for grpc

from __future__ import print_function
import logging

import numpy as np
import base64

import grpc

import communication_pb2
import communication_pb2_grpc


##########################
# Server-Port selection  #
##########################
# Local use:
#srv_port="127.0.1.1:50051"

# Docker use:
srv_port="172.17.0.1:50051"
      
      
##########################
# GRPC-Client definition #
##########################
class GRPCClient:
    """
    Class to control a GRPC client.
    
    Attributes
    ----------
    client_id      : unique identifier for the client instance
    channel        : network and port where the communication will be established
    stub           : connection unit to channel
    ready          : flag - client waiting for commands
    """

    # own functions
    def __init__(self, client_id=1, server_port=srv_port):
        self.client_id = client_id
        self.server_port = server_port
        
        self.channel = grpc.insecure_channel(self.server_port)
        self.stub = communication_pb2_grpc.GRPCServerStub(self.channel)
        
        self.ready = True
        
    def test(self):
        """
        Test if connection is ok.
        
        return:
          - server_id
        """
        # send test request
        response = self.stub.Test(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.server_id
        
    def get_client_id(self):
        return self.client_id
        
    def get_server_port(self):
        return self.server_port
        
        
    # Experiment wrapper
    def setup(self):
        """
        Get position of target cylinder from simulation.
        
        return:
          - cylinder_position 
        """        
        # send start signal to simulation
        response = self.stub.Setup(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return [response.pos_x, response.pos_y, response.pos_z]
        
    def execute(self, act):
        """
        Send action to simulation.
        
        return:
          - distance to target cylinder 
        """
        # send action to simulation
        response = self.stub.Execute(communication_pb2.RequestDoAction(
            client_id=self.client_id,
            joint_1=act[0],
            joint_2=act[1],
            joint_3=act[2],
            joint_4=act[3],
            joint_5=act[4],
            joint_6=act[5]
            ))
        return response.distance
        
    def distance_robot_cylinder(self):
        """ 
        Get the distance between all joints and cylinder 

        return:
           - distance between the last joint (end effector) an cylinder
        """
        # send action to simulation
        response = self.stub.DistanceRobotCylinder(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.distance
        
        
    # Robot wrapper
    def robot_reset(self):
        """
        Resets robot joints in simulation to the initial state.
        
        return: 
          - True/False (if response appeared/ if no response appeared)
        """
        # reset robot in simulation
        response = self.stub.RobotReset(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return True if response.server_id else False
    
    def get_robot_joint_states(self):
        """
        Get joint states of robot from simulation.
        
        return:
          - robot joint states 
        """
        # send rjs request to simulation
        response = self.stub.GetRobotJointStates(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return [response.state_joint_1, response.state_joint_2, response.state_joint_3, 
                response.state_joint_4, response.state_joint_5, response.state_joint_6]
        
    #def get_current_frame(self):
        
    def get_robot_position(self):
        """
        Get  position of robot from simulation.
        
        return:
          - robot position
        """
        # send position request to simulation
        response = self.stub.GetRobotPosition(communication_pb2.RequestEmpty(
            client_id=self.client_id,
            ))
        return (
            [response.joint_1_x, response.joint_1_y, response.joint_1_z], 
            [response.joint_3_x, response.joint_3_y, response.joint_3_z],
            [response.joint_5_x, response.joint_5_y, response.joint_5_z], 
            [response.ee_x, response.ee_y, response.ee_z]
        )
        
    def robot_act(self, act):
        """
        Send action to robot in simulation.
        
        return:
          - True/False (if response appeared/ if no response appeared) 
        """
        # send action to robot
        response = self.stub.RobotAct(communication_pb2.RequestDoAction(
            client_id=self.client_id,
            joint_1=act[0],
            joint_2=act[1],
            joint_3=act[2],
            joint_4=act[3],
            joint_5=act[4],
            joint_6=act[5]
            ))
        return True if response.server_id else False
        
    def robot_is_stable(self):
        """
        Tests if robot in simulation is stable.
        
        return:
          - is_stable (binary)
        """
        # send request to simulation
        response = self.stub.RobotIsStable(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.status
        
    def robot_check_collision(self):
        """
        Tests if robot in simulation did collide.
        
        return:
          - collision (binary)
        """
        # send request to simulation
        response = self.stub.RobotCheckCollision(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.status
        
        
    # Cylinder wrapper
    def cylinder_reset(self):
        """
        Resets cylinder position in simulation to the initial state.
        
        return: 
          - True/False (if response appeared/ if no response appeared)
        """
        # reset cylinder in simulation
        response = self.stub.CylinderReset(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return True if response.server_id else False
        
    def cylinder_random_reset(self):
        """
        Resets cylinder position in simulation to random state.
        
        return: 
          - True/False (if response appeared/ if no response appeared)
        """
        # reset cylinder in simulation
        response = self.stub.CylinderRandomReset(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return True if response.server_id else False

    def cylinder_get_position(self):
        """
        Get position of cylinder from simulation.
        
        return:
          - cylinder_position 
        """   
        # get cylinder position from simulation
        response = self.stub.CylinderGetPosition(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return [response.pos_x, response.pos_y, response.pos_z]
        
    def cylinder_is_stable(self):
        """
        Tests if cylinder in simulation is stable.
        
        return:
          - is_stable (binary)
        """
        # send request to simulation
        response = self.stub.CylinderIsStable(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.status
        
    def cylinder_is_on_ground(self):
        """
        Tests if cylinder in simulation is on ground
        
        return:
          - on_ground (binary)
        """
        # send request to simulation
        response = self.stub.CylinderIsOnGround(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        return response.status
        
    # Camera wrapper
    def camera_get_image(self):
        """
        Gets the camera image.
        
        return: 
          - camera image (np.ndarray)
        """
        # send request to simulation
        response = self.stub.CameraGetImage(communication_pb2.RequestEmpty(
            client_id=self.client_id
            ))
        b64decoded = base64.b64decode(response.b64image)
        img = np.frombuffer(b64decoded, dtype=np.uint8).reshape(response.width, response.height, -1)
        return img

