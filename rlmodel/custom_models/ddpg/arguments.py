#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import math

"""
Here are the parameters for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting, which could be used in the command line

    '''
    python demo.py -h
    '''
    parser.add_argument('--env-name', type=str,
                        default='HandManipulateBlockRotateZ-v0', help='the environment name')  
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40,
                        help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str,
                        default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float,
                        default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str,
                        default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float,
                        default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float,
                        default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int,
                        default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4,
                        help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float,
                        default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001,
                        help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001,
                        help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95,
                        help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int,
                        default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float,
                        default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int,
                        default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true',
                        help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int,
                        default=2, help='the rollouts per mpi')

    args = parser.parse_args()

    return args


class Args:
    def __init__(self):

        # or "image" (not implemented)
        self.train_type = "trueData" 

        # just a random seed 123
        self.seed = 125  

        # True
        self.cuda = False 
        '''
        During running, the following error was found. At last cuda wasn't used.
        RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU 
        and installed a driver from http://www.nvidia.com/Download/index.aspx
        '''

        # actor
        self.lr_actor = 0.001 

        # critic
        self.lr_critic = 0.001 
        
        # future can get better results.
        self.replay_strategy = 'future'  

        # replay with k random states which come from the same episode as the transition being replayed and were observed after it
        self.replay_k = 4  

        # can be adjusted dynamically
        self.buffer_size = 1e6*1/2  

        # add demo data or not, not used actually 
        self.add_demo = False  

        # math.pi means all joint can move from -pi to pi
        self.clip_range = math.pi 

        # path to save the models
        self.save_dir = 'saved_models/' 

        # define the name of file
        self.env_name = 'robot_'+str(self.train_type)+" seed"+str( self.seed )
        
        # define how many epochs you want to run
        self.n_epochs = 200

        # define how many cycles in each epoch you want to run
        self.n_cycles = 50 

        # define the number of rollouts
        self.num_rollouts_per_mpi = 2

        # define hyperparameter
        self.noise_eps = 0.01

        self.random_eps = 0.3

        # define number of batchs while training the agent
        self.n_batches = 40 

        # l2-regularization while training the network
        self.action_l2 = 1 

        # how many times you want to evaluation
        self.n_test_rollouts = 25    

        # after this number of training, some "big actions" will be deleted
        self.clip_obs = 200

        # define batch size
        self.batch_size = 256

        # define hyperparameter
        self.gamma = 0.98

        # soft update
        self.polyak = 0.95  
