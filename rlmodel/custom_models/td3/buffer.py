import numpy as np

import torch

class ReplayBuffer(object):
    """ A simple FIFO experience replay buffer for TD3 agents."""
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """
            Args:
                state_dim (int): number of elements in the state space
                action_dim (int): number of elements in the action space
                max_size (int): total amount of tuples to store
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def store(self, state, action, next_state, reward, done):
        """ Add experience tuples to buffer
        
            Args:
                state: observation of the current environment state
                action: action taken in the current state 
                next_state: the reached state after taking a given action in a given state
                reward (float): reward for taking a given action in a given state (aims to show how useful the action was)
                done (boolean): whether it’s time to reset the environment again. done being True indicates the episode has terminated.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        """ Samples a random amount of experiences from buffer of batch size
        
            Args:
                batch_size (int): size of sample
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )