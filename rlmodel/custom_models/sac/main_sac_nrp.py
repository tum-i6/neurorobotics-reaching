import gym
import numpy as np
from sac_agent import Agent
from gym import wrappers, spaces
from itertools import zip_longest
import sys
import os

# Setup are GRPC
sys.path.insert(1, os.path.join(sys.path[0], '/tum_nrp/grpc/python/communication'))
import communication_client as cc

# YACS setup config
from config import get_cfg_defaults  

# Setup tensorboard 
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    ###### Load YACS config ######
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./configs/base_setup.yaml")
    cfg.freeze()
    print(cfg)

    ###### Setup Tensorboard ######
    writer = SummaryWriter('tensorboard_log/',comment="-SAC") # set up tensorboard storage

    ###### Setup and test GRPC ######

    print("Instanciate a GRPC-client...", end='')
    client = cc.GRPCClient()
    print('\033[92m' + "done" + '\033[0m')

    print("Make a test request...", end='')
    ok = client.test()
    print('\033[92m' + "done" + '\033[0m' + " result: " + str(ok))

    print("Make a second test request...", end='')
    ok = client.test()
    print('\033[92m' + "done" + '\033[0m' + " result: " + str(ok))

    ###### Setup the action and observation spaces #######

    # Max and min  value of the angles of the robot joints
    rob_h = cfg.SPACES.ROB_ANGLE_HIGH
    rob_l = cfg.SPACES.ROB_ANGLE_LOW 

    # Observation space of the cylinder (x, y, z, rotation_x, rotation_y, rotation_z, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6)
    observation_space = spaces.Box(low=np.array([cfg.SPACES.CYLINDER_X_LOW, cfg.SPACES.CYLINDER_Y_LOW, cfg.SPACES.CYLINDER_Z_LOW, rob_l/2, 0, 0]), 
                                    high=np.array([cfg.SPACES.CYLINDER_X_HIGH, cfg.SPACES.CYLINDER_Y_HIGH, cfg.SPACES.CYLINDER_Z_HIGH, rob_h/2, rob_h/2, rob_h]), dtype=np.float32)
    # Action Space
    action_space = spaces.Box(low=np.array([rob_l/2, 0, 0, 0, 0, 0]), 
                                high=np.array([rob_h/2, rob_h/2, rob_h, 0,0,0]), dtype=np.float32)

    ###### Initiate the Agent and setup the experiment ######

    agent = Agent(alpha=cfg.AGENT.ALPHA, beta=cfg.AGENT.BETA, input_dims=observation_space.shape, action_space=action_space, gamma=cfg.AGENT.GAMMA, n_actions=action_space.shape[0], 
                max_size=cfg.AGENT.MAX_SIZE, tau=cfg.AGENT.TAU, layer1_size=cfg.AGENT.LAYER_ONE, layer2_size=cfg.AGENT.LAYER_TWO, batch_size=cfg.AGENT.BATCH_SIZE, reward_scale=cfg.AGENT.REWARD_SCALE)

    # Action space of the robot joint_1, joint_2, joint_3

    # Number of runs/games
    epochs = cfg.TRAIN.NR_GAMES

    # Error value to avoid devision through zero
    eps = sys.float_info.epsilon

    # We only allow one action per round 
    single_action_per_round = cfg.TRAIN.SINGLE_ACTION_PER_ROUND 

    if not single_action_per_round:
        max_ran_per_episode = cfg.TRAIN.MAX_ROUNDS_PER_EPISODE
    else:
        max_ran_per_episode = 1

    # Define the reward range
    reward_range = (cfg.TRAIN.MIN_REWARD, cfg.TRAIN.MAX_REWARD)
    best_score = reward_range[0]

    # Setup score recording
    score_history = []

    # Load Checkpoint
    load_checkpoint = cfg.TRAIN.LOAD_CHECKPOINT

    if load_checkpoint:
        print("Loading Checkpoint")
        agent.load_models()

    ###### Start experiment ######

    for i in range(epochs):
        print("#"*20, " Game Nr.: ", i, " ", "#"*20)
        
        print("Reset the simulation..." , end='')
        cylinder_pos = client.setup() 
        print('\033[92m' + "done" + '\033[0m')

        # Get the position of the end-effector and the robot jo int states
        print("Retrive the the postion of the end-effector..." , end='')
        end_effector_pos = client.get_robot_position()[-1]
        print('\033[92m' + "done" + '\033[0m')

        print("Get the observation..." , end='')
        observation = cylinder_pos + end_effector_pos
        print('\033[92m' + "done" + '\033[0m')
        
        # To calculate a meaningful reward we need to know how far away the end effector and cylinder are at setup
        starting_distance = client.execute([0,0,0,0,0,0])

        done = False
        score = 0

        cur_run_per_episode = 0
        max_ran_per_episode = 2
        while not done:            
            print("Select and execute an action in the simulation..." , end='')
            action = agent.choose_action(observation)
            distance_robot_cylinder = client.execute(action)
            print('\033[92m' + "done" + '\033[0m')
            
            # Keep the cylinder position but update the joint angles by adding the action to the joint values
            end_effector_pos = client.get_robot_position()[-1]
            observation_ = cylinder_pos +end_effector_pos
            
            # Reward: Increase the reward expoentialy if the robot gets closer to the cylinder, cut of the reward curve at 15 points 
            reward= min(pow((starting_distance/distance_robot_cylinder),2)-1, 15)
    
            # Only single action per episode, max number of actions, or until the distance between the robot and the cylinder < x
            if single_action_per_round or cur_run_per_episode >= max_ran_per_episode:
                done = True
            elif distance_robot_cylinder < 0.2:
                done = True
                # Increase reward if the robot crosses - dependent on the tries needed
                # Plus one to avoid devision by zero
                reward = reward + (30/(cur_run_per_episode+1))
            else:
                done = False

            # The max distance between the cylinder and the robot is ca. 2.5 while the min is 0
            score += reward
            
            # Update the replay buffer
            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()
                
            # Update the observation                
            observation = observation_

            # Increment the counter for runs per episode
            cur_run_per_episode += 1
            print('episode ', i, 'run ', cur_run_per_episode, 'episode_score %.3f' % score, 'current_run_score %.3f' % reward)

        # Update the score history with the score from the last episode        
        score_history.append(score)

        # Calculate the avg score        
        avg_score = np.mean(score_history[-100:])
        
        # Save the model state if the average score is higher then the best score     
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        # Update Tensorboard
        writer.add_scalar("avg_reward", avg_score, i)
        writer.add_scalar("score", score, i)


        print('episode ', i, 'score %.3f' % score, 'avg_score %.3f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
