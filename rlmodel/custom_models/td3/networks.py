import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rescale

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    """ The “Actor” updates the policy distribution in the direction suggested by the Critic.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            min_action (array or float): lowest action to take
            max_action (array or float): highest action to take
            
        Return:
            action output of network with tanh activation
    """
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.min_action = min_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) # range: [-1,1]
        return rescale(input=a, out_range=(self.min_action, self.max_action), in_range=(-1, 1))         
   

class Critic(nn.Module):
    """ The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value 
        (the V value).

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            
        Return:
            value output of network 
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    """ Agent class that handles the training of the networks and provides outputs as actions
    
        Args:
            state_dim (int): state size
            action_dim (int): action size
            min_action (array or float): lowest action to take
            max_action (array or float): highest action to take
            discount (float): discount factor
            tau (float): soft update for main networks to target networks
            policy_noise (float): Amplitude of noise to be added on each dimension of the action (target policy)
            noise_clip (float): Amplitude for noise clipping in order to keep the target value close to the original action
            policy_freq (int): Update the policy every policy_freq iterations to increase policy stability
            lr (int): learning rate to be used for the training of the models
    
    """
    
    def __init__(self, state_dim, action_dim, max_action, min_action, discount=0.99, 
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, lr=3e-4):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = torch.from_numpy(max_action)
        self.min_action = torch.from_numpy(min_action)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = torch.from_numpy(noise_clip)
        self.policy_freq = policy_freq
        
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, self.min_action).to(self.device)
        self.actor.apply(init_weights)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic.apply(init_weights)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.total_it = 0
        
    def select_action(self, state, expl_noise = 0.1):
        """ Select an appropriate action from the agent policy
        
            Args:
                state (array): current state of environment
                expl_noise (float): Standard deviation (spread or “width”) of the distribution from which the action noise samples are drawn. Must be non-negative.
                
            Returns:
                action (float): action clipped within action range
        
        """
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)        
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if expl_noise != 0:
            expl_noise = (self.max_action - self.min_action) / 2 * expl_noise
            action = (action + np.random.normal(0, expl_noise, size=self.action_dim))
            
        return action.clip(self.min_action, self.max_action)
    
    def train(self, replay_buffer, batch_size=256):
        """ Do one iteration of actor and critic network training and updating.
        
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                batch_size(int): batch size to sample from replay buffer
                
        """
        
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():            
            # Select action according to policy and add clipped noise 
            noise = (torch.randn_like(action) * self.policy_noise)#.clamp(-self.noise_clip, self.noise_clip)
            noise = torch.where(noise > self.noise_clip, self.noise_clip, noise) # torch.clamp() does not take Tensors as min/max arguments
            noise = torch.where(noise < -self.noise_clip, -self.noise_clip, noise) 

            next_action = (self.actor_target(next_state) + noise)#.clamp(self.min_action, self.max_action)
            next_action = torch.where(next_action > self.max_action, self.max_action, next_action) # torch.clamp() does not take Tensors as min/max arguments
            next_action = torch.where(next_action < self.min_action, self.min_action, next_action) 
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
             
    def save(self, filename, directory):
        """ Save actor and critic models to a given location

            Args:
                filename (string): Name of the TD3 agent models to save
                directory (string): Directory under which to save the models
        """
        torch.save(self.actor.state_dict(), '%s/%s_actor.pt' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pt' % (directory, filename))
        
        torch.save(self.critic.state_dict(), '%s/%s_critic.pt' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pt' % (directory, filename))


    def load(self, filename="best_avg", directory="./saves"):
        """ Load actor and critic models from a given location

            Args:
                filename (string): Name of the TD3 agent models to load
                directory (string): Directory from which to load the models
        """
        self.actor.load_state_dict(torch.load('%s/%s_actor.pt' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pt' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic.load_state_dict(torch.load('%s/%s_critic.pt' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pt' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)


def init_weights(m):
    """ Apply proper weight initialization to the deep learning models to facilitate the training process and reduce vanishing/exploding gradient problems.
        For more information see: https://pytorch.org/docs/stable/nn.init.html
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)