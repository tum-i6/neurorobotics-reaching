print("Some simple model example to show how to use GRPC-Client.")

import random
import time

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../grpc/python/communication'))

import experiment_api_wrapper as eaw

# instanciate a experiment (wrapper)
print("Instanciate a GRPC-client...", end='')
experiment = eaw.ExperimentWrapper()
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Make a test request...", end='')
server_id = experiment.client.test()
print('\033[92m' + "done" + '\033[0m' + "\n" + "server_id: " + str(server_id) + "\n")


print("Setup the simulation..." , end='')
pos = experiment.setup()
print('\033[92m' + "done" + '\033[0m')
print("Cylinder position is:\n", pos)
print("\n")


# create some random action
action = [0] * 6
for i in range(0, 5):
    action[i] = random.uniform(0, 1)

print("Randomly generated action:\n", action)
print("Execute action in simulation..." , end='')
pos = experiment.execute(action)
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Request distance to cylinder..." , end='')
dist = experiment.distance_robot_cylinder()
print('\033[92m' + "done" + '\033[0m')
print("Distance is:\n", dist)
print("\n")


print("Reset robot to initial position..." , end='')
ok = experiment.robot.reset()
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Request the robot joint states...", end='')
rjs = experiment.robot.get_joint_states()
print('\033[92m' + "done" + '\033[0m')
print("Robot joint states are:\n", rjs)
print("\n")


print("Request the robot position...", end='')
rpos = experiment.robot.get_position()
print('\033[92m' + "done" + '\033[0m')
print("Robot position is:\n", rpos)
print("\n")


# create some random action
action = [0] * 6
for i in range(0, 5):
    action[i] = random.uniform(0, 1)

print("Randomly generated action:\n", action)
print("Execute action in simulation..." , end='')
pos = experiment.robot.act(action)
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Request robot state (is_stable)...", end='')
stable = experiment.robot.is_stable()
print('\033[92m' + "done" + '\033[0m' + "\n" + "stable: " + str(stable)+ "\n")


print("Check if robot did collide...", end='')
collision = experiment.robot.check_collision()
print('\033[92m' + "done" + '\033[0m' + "\n" + "collision: " + str(collision)+ "\n")


print("Reset cylinder...", end='')
ok = experiment.cylinder.reset()
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Reset cylinder randomly...", end='')
ok = experiment.cylinder.random_reset()
print('\033[92m' + "done" + '\033[0m' + "\n")


print("Get cylinder position...", end='')
pos = experiment.cylinder.get_position()
print('\033[92m' + "done" + '\033[0m' + "\n" + "position: " + str(pos) + "\n")


print("Request cylinder state (is_stable)...", end='')
stable = experiment.cylinder.is_stable()
print('\033[92m' + "done" + '\033[0m' + "\n" + "stable: " + str(stable) + "\n")


print("Request cylinder state (is_on_ground)...", end='')
ground = experiment.cylinder.is_on_ground()
print('\033[92m' + "done" + '\033[0m' + "\n" + "on ground: " + str(ground) + "\n")

print("Request camera image...", end='')
img = experiment.camera.get_image()
print('\033[92m' + "done" + '\033[0m' + "\n")
print("img.shape: ", img.shape)
print("img: ", img) 

