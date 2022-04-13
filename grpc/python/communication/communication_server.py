# Author: Kevin Farkas (5.Mai 2021)
#
# Content:
# - simple communciation-server for grpc

import sys
import os
import numpy as np
import base64
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../experiment'))

from concurrent.futures import ThreadPoolExecutor
import logging

import grpc

import communication_pb2
import communication_pb2_grpc


# ServerID: fixed due to single server 
SERVER_ID = 1


##########################
# Simulation settings    #
##########################
import experiment_api
experiment = experiment_api.Experiment()


##########################
# Server-Port settings   #
##########################
from ip_address import automate_ip_address
server_port = automate_ip_address.get_grpc_server_port()
#server_port = '127.0.1.1:50051'

##########################
# GRPC-Server definition #
##########################
class GRPCServer(communication_pb2_grpc.GRPCServerServicer):
    """
    Class to control a GRPC server.
    
    Attributes
    ----------
    server_id      : unique identifier for the client instance
    
    """
    # Experiment
    def __init__(self): #server
        self.server_id = SERVER_ID
        print("GRPC-server-port: ", server_port)
        
    def Test(self, request, context):
        """
        Service that tests the connection.
        
        return:
        - empty response
        """
        return communication_pb2.ResponseEmpty(
            server_id=SERVER_ID
            )
        
    def Setup(self, request, context):
        """
        Service that setups the simulation.
        
        return:
        - cylinder_position
        """
        # call simulation
        pos = experiment.setup()
        # response
        return communication_pb2.ResponseCylinderPos(
            server_id=SERVER_ID, 
            pos_x=pos[0],
            pos_y=pos[1],
            pos_z=pos[2]
            )
        
    def Execute(self, request, context):
        """
        Service that executes action in the simulation.
        
        return:
        - distance to target cylinder
        """
        # call simulation
        dist = experiment.execute(
            request.joint_1, 
            request.joint_2, 
            request.joint_3, 
            request.joint_4,
            request.joint_5,
            request.joint_6
            )
        # response
        return communication_pb2.ResponseDistance(
            server_id=SERVER_ID, 
            distance=dist
            )
            
    def DistanceRobotCylinder(self, request, context):
        """
        Service that gets distance between robot and cylinder in the simulation.
        
        return:
        - distance between robot and cylinder
        """
        # call simulation
        dist = experiment.distance_robot_cylinder()
        # response
        return communication_pb2.ResponseDistance(
            server_id=SERVER_ID, 
            distance=dist
            )
            
            
    # Robot
    def RobotReset(self, request, context):
        """
        Service that resets robot in simulation to initial state.
        
        return:
        - empty response
        """
        # call simulation
        experiment.robot.reset()
        # response
        return communication_pb2.ResponseEmpty(
            server_id=SERVER_ID
            )
    
    def GetRobotJointStates(self, request, context):
        """
        Service that gets the current robot joint states from simulation.
        
        return:
        - robot joint states
        """
        # call simulation
        rjs = experiment.robot.get_joint_states()
        # response
        return communication_pb2.ResponseRobotJointStates(
            server_id=SERVER_ID,
            state_joint_1=rjs[0],
            state_joint_2=rjs[1],
            state_joint_3=rjs[2],
            state_joint_4=rjs[3],
            state_joint_5=rjs[4],
            state_joint_6=rjs[5]
            )
            
    def GetRobotPosition(self, request, context):
        """
        Service that gets the current robot position from simulation.
        
        return:
        - robot position
        """
        # call simulation
        rp = experiment.robot.get_position()
        # response
        return communication_pb2.ResponseRobotPosition(
            server_id=SERVER_ID,
            joint_1_x=rp[0][0],
            joint_1_y=rp[0][1],
            joint_1_z=rp[0][2],
            joint_3_x=rp[1][0],
            joint_3_y=rp[1][1],
            joint_3_z=rp[1][2],
            joint_5_x=rp[2][0],
            joint_5_y=rp[2][1],
            joint_5_z=rp[2][2],
            ee_x=rp[3][0],
            ee_y=rp[3][1],
            ee_z=rp[3][2]
            )
            
    def RobotAct(self, request, context):
        """
        Service that forwards an action to robot in the simulation.
        
        return:
        - empty response
        """
        # call simulation
        dist = experiment.robot.act(
            request.joint_1, 
            request.joint_2, 
            request.joint_3, 
            request.joint_4,
            request.joint_5,
            request.joint_6
            )
        # response
        return communication_pb2.ResponseDistance(
            server_id=SERVER_ID, 
            distance=dist
            )
            
    def RobotIsStable(self, request, context):
        """
        Service that gets status of robot.
        
        return:
        - robot status
        """
        # call simulation
        stable = experiment.robot.is_stable()
        # response
        return communication_pb2.ResponseBinaryStatus(
            server_id=SERVER_ID,
            status = stable
            )
            
    def RobotCheckCollision(self, request, context):
        """
        Service that checks if robot did collide.
        
        return:
        - collision 
        """
        # call simulation
        collision = experiment.robot.check_collision()
        # response
        return communication_pb2.ResponseBinaryStatus(
            server_id=SERVER_ID,
            status = collision
            )


    # Cylinder
    def CylinderReset(self, request, context):
        """
        Service that resets cylinder in simulation to initial state.
        
        return:
        - empty response
        """
        #call simulation
        experiment.cylinder.reset()
        return communication_pb2.ResponseEmpty(
            server_id=SERVER_ID
            )
            
    def CylinderRandomReset(self, request, context):
        """
        Service that resets cylinder in simulation to random state.
        
        return:
        - empty response
        """
        # call simulation
        experiment.cylinder.random_reset()
        return communication_pb2.ResponseEmpty(
            server_id=SERVER_ID
            )
            
    def CylinderGetPosition(self, request, context):
         """
         Service that returns the current cylinder position in simulation.
         
         return:
         - cylinder_position
         """
         # call simulation
         pos = experiment.cylinder.get_position()
         return communication_pb2.ResponseCylinderPos(
             server_id=SERVER_ID, 
             pos_x=pos[0],
             pos_y=pos[1],
             pos_z=pos[2]
             )
             
    def CylinderIsStable(self, request, context):
        """
        Service that gets status of cylinder.
        
        return:
        - cylinder status
        """
        # call simulation
        stable = experiment.cylinder.is_stable()
        # response
        return communication_pb2.ResponseBinaryStatus(
            server_id=SERVER_ID,
            status=stable
            )
            
    def CylinderIsOnGround(self, request, context):
        """
        Service that gets status of cylinder.
        
        return:
        - cylinder status
        """
        # call simulation
        on_ground = experiment.cylinder.is_on_ground()
        # response
        return communication_pb2.ResponseBinaryStatus(
            server_id=SERVER_ID,
            status=on_ground
            )
     
     
    # Cameras
    def CameraGetImage(self, request, context):
        """
        Service that gets the camera image.
        
        return:
        - camera image
        """
        # call simulation
        image = experiment.cameras.get_image() 
        # form numpy.ndarray with shape (HxWxC) to serialized string
        data = base64.b64encode(image)
        # width and height
        w, h, _ = image.shape
        # response
        return communication_pb2.ResponseCameraImage(
            server_id=SERVER_ID,
            b64image=data,
            width=w,
            height=h
            )
        
         

##########################
# run server instance    #
##########################
def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_GRPCServerServicer_to_server(GRPCServer(), server)
    server.add_insecure_port(server_port)
    server.start()
    server.wait_for_termination()
    
if __name__ == '__main__':
    logging.basicConfig()
    serve() 

