""" 
Custom TD3 Model

This is the main function to create a TD3 agent, communicate with the Neurorobotics Platform environment
to execute reaching episodes, train and evaluate the agent based on the numeric data from the environment
with different settings and log all data to tensorboard. 
"""
__author__ = 'Marton Szep'

import sys
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(1, '/tum_nrp/grpc/python/communication')
import experiment_api_wrapper as eaw

from buffer import ReplayBuffer
from networks import TD3
from train_helpers import Environment, Runner, evaluate_policy, observe, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2e3, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=10e3, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.1, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.25, type=float)   # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--lr", default=3e-4, type=float)           # Learning rate for network update
    parser.add_argument("--proximity", default=0.2, type=float)     # Goal proximity of robot end effector and target
    parser.add_argument("--reward_thresh", default=0.9, type=float) # Goal reward threshold to finish model training
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load
    parser.add_argument("--eval_only", action="store_true")         # Do not train, just evaluate model from default folder
    parser.add_argument("--eval_episodes", default=100, type=int)   # Number of episodes to evaluate in an evaluation step.
    parser.add_argument("--tries_per_episode", default=1, type=int) # How many times in an epsiode can the agent try to reach the goal
    parser.add_argument("--fixed_goal", action="store_true")        # Fix the target position throughout the training
    parser.add_argument("--use_3_joints", action="store_true")      # Use only the first 3 joints to try to solve the robotic reaching task
    parser.add_argument("--comment", default="")                    # Comment to be added to the filename
    args = parser.parse_args()
  
    if args.comment !="":
        file_name = f"TD3_{args.proximity}_{args.tries_per_episode}_{args.start_timesteps}_{args.max_timesteps}_{args.comment}"
    else:
        file_name = f"TD3_{args.proximity}_{args.tries_per_episode}_{args.start_timesteps}_{args.max_timesteps}"

    print("---------------------------------------")
    print(file_name)
    print("---------------------------------------")
    
    if not args.eval_only:
        writer = SummaryWriter(f"tensorboard_log/{file_name}/",comment="-TD3") # set up tensorboard storage

    params = {
        "SEED": args.seed,
        "OBSERVATIONS": args.start_timesteps,
        "EXPLORATION": args.max_timesteps,
        "BATCH_SIZE": args.batch_size,
        "GAMMA": args.discount,
        "TAU": args.tau,
        "POLICY_NOISE": args.policy_noise,
        "NOISE_CLIP": args.noise_clip,
        "EXPLORE_NOISE": args.expl_noise,
        "POLICY_FREQUENCY": args.policy_freq,
        "EVAL_FREQUENCY": args.eval_freq,
        "REWARD_THRESH": args.reward_thresh,
        "LR": args.lr,
        "DISTANCE_THRESH": args.proximity,
        "TRIES": args.tries_per_episode,
        "EVAL_EPISODES": args.eval_episodes,
        "FILENAME": file_name}

    torch.manual_seed(params["SEED"])
    np.random.seed(params["SEED"])
    
    # instanciate a grpc-client
    print("Instanciate a GRPC-client...", end='')
    experiment = eaw.ExperimentWrapper()
    print('\033[92m' + "done" + '\033[0m')

    print("Make a test request...", end='')
    ok = experiment.client.test()
    print('\033[92m' + "done" + '\033[0m' + " result: " + str(ok))

    print("Make a second test request...", end='')
    ok = experiment.client.test()
    print('\033[92m' + "done" + '\033[0m' + " result: " + str(ok)+ "\n")

    # Instanciate environment wrapper class for interaction in the simulation.
    env = Environment(experiment, threshold = params["DISTANCE_THRESH"], random_goal=(not args.fixed_goal), 
                    use_3_joints=args.use_3_joints)
    print("fixed_goal: ", args.fixed_goal)
    
    print("3 joints used: ", args.use_3_joints)
    state_dim = env.observation_space.shape[0]
    print('state_dim: ', state_dim)
    action_dim = env.action_space.shape[0]
    print('action_dim: ', action_dim)
    min_action = env.action_space.low
    print('min_action:', min_action)
    max_action = env.action_space.high
    print('max_action:', max_action)
    print('Goal proximity: {}\n'.format(args.proximity))

    # Instanciate TD3 agent
    noise_clip=params["NOISE_CLIP"]*(max_action - min_action) / 2
    policy_noise=params["POLICY_NOISE"]*(max_action - min_action) / 2
    policy = TD3(state_dim, action_dim, max_action, min_action, discount=params["GAMMA"], tau=params["TAU"], 
                policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=params["POLICY_FREQUENCY"], 
                lr=params["LR"])
    

    # Load model if argument given.
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(filename=f"{policy_file}", directory="./saves") 
        print("Model has been loaded: {}".format(policy_file)) 


    # Train the model if not stated otherwise.   
    if not args.eval_only: 
        # Instanciate experience buffer and runner for the training process.
        replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        runner = Runner(env, policy, replay_buffer)

        # Populate replay buffer
        observe(env, replay_buffer, params["OBSERVATIONS"], tries=params["TRIES"])

        # Train agent
        train(policy, env, runner, replay_buffer, params, writer)

    # Evaluate best policy
    if not args.eval_only: 
        print("Final evaluation of best performing model: ")
        policy.load(filename=file_name, directory="./saves")
    else:
        print("Model evaluation for {} episodes has started.".format(params["EVAL_EPISODES"]))
        writer = SummaryWriter(f"tensorboard_log/{policy_file}/",comment="-TD3") # set up tensorboard storage
    avg_reward, avg_success, avg_distance, fig = evaluate_policy(policy, env, eval_episodes=params["EVAL_EPISODES"], tries=params["TRIES"], plot_scatter=True)
    writer.add_figure(f"Model performance evaluation (threshold: {params['DISTANCE_THRESH']}m)", fig)

    if not args.eval_only:       
        writer.add_hparams(params,{"avg_reward":avg_reward, "avg_success": avg_success, "avg_distance": avg_distance})
    writer.close()