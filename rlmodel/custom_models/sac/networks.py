import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        """ Initializes the critic network. The critic network takes a state and an action as input and tells the 
            actor network if the taken action was good or bad.

            Args:
                beta: Learning rate
                input_dims: Input dimension
                n_action: Number of components of the action (since continour env.)
                fc1_dims: Dimmension of the first fully connected layer (default form the paper)
                fc2_dims: Dimmension of the second fully connected layer (default form the paper)
                name: Network identifier
                chkpt_dir: Checkpoint directory
        """
        super(CriticNetwork, self).__init__()

        # Save the parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        # Carfull: This directory has to exist before the program is run
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Init the two fully connected layers
        # The critic evaluates the value of the state and action pair
        # We therefore pass the action along through the network
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Get the q value (scalar quantity)
        self.q = nn.Linear(self.fc2_dims, 1)

        # Optimizer 
        # Optimize the NN's parameters 
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Use GPU if avilabe
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Send the whole network to the device
        self.to(self.device)

    def forward(self, state, action):
        """ Forward pass through the model
            
            Args:
                state: The state
                action: The action
            
            Returns: 
                The value of the action given the current state
        """
        # The action value is the feedforward of the concatenation of the state and action along the batch dimension 
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        """ Save the state dictionary of the model
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """ Load the state dictionary
        """
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        """ Initializes the value network. The value network evalutes if a state is valuable or not - it does not care about the action you take or toke.
            Takes as input simply the state.
            Args:
                beta: Learning rate
                input_dims: Input dimension
                fc1_dims: Dimmension of the first fully connected layer (default form the paper)
                fc2_dims: Dimmension of the second fully connected layer (default form the paper)
                name: Network identifier
                chkpt_dir: Checkpoint directory
        """
        super(ValueNetwork, self).__init__()

        # Save the parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Basic NN
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        # Setup the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Pass the network to the device
        self.to(self.device)

    def forward(self, state):
        """ Forward pass through the model

            Args:
                state: The state

            Return:
                The value of the current state
        """
        # The value is the feedforward of the state through the network
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        """ Save the state dictionary of the model
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """ Load the state dictionary
        """
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        """ Initializes the actor network. The actornetwork handels the policy - handles how the agent selects an action. 
            The network outputs a mean and standart deviation. It has to sample the propability distribution
            Args:
                alpha: Learning rate
                input_dims: Input dimension
                max_action: The sampling function is restricted to a range of -1 to +1. We multiply with max_action to get task specific values 
                fc1_dims: Dimmension of the first fully connected layer (default form the paper)
                fc2_dims: Dimmension of the second fully connected layer (default form the paper)
                name: Network identifier
                chkpt_dir: Checkpoint directory
        """
        super(ActorNetwork, self).__init__()
        # Save the parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        # Reparameterization: Ensures that we e.g. do not take the log of zero
        self.reparam_noise = 1e-6

        # Two fully connected layers through which we pass the state
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        # Two output layers, one for the mu and one for the sigma value of the distribution
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        # Setup the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """ Forward pass through the model
            
            Args:
                state: The state

            Return:
                mu: The mu of the standard distribution
                sigma: The sigma of the standard distribution
        """
        # The state value is the feedforward through the fully connected layers
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        # The output of the second fully connected layer is passed to the mu and sigma layer respectively
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # Clamp the sigma to make sure that the spread of the distribution is limited
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """ Handels the actual calculation of the policy.
            (The policy is a proverbiality distribution that tells the proverbiality of selecting an action given some state or set of state)
        
            Args:
                state: The state
                reparameterize: Flag that introduces noise if set to true

            Return:
                Action: The sampled action that should be taken by the agent
                log_probs: Loss value for updating the weight of the NN
        """
        # Get the normal distribution
        mu, sigma = self.forward(state)

        probabilities = Normal(mu, sigma)

        # Sample form above distribution, either with or without noise
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # The action for the agent is then the actions passed to a tanh activation
        # multiplied by the max action value and passed to the device (all tensors have to be on the same device)
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)

        # Calculating the loss value for updating the weights of the NN
        # Very important: We wan to use actions here and not action since action is proportional to the tanh and the max action value
        log_probs = probabilities.log_prob(actions)
        # Substract of the log one minus the square of the action + parameter noise (avoid log of zero)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise) 
        # Take the sum as we need a scalar quantity (privies dimensionality is corresponding to the number of components of the actions)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        """ Save the state dictionary of the model
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """ Load the state dictionary
        """
        self.load_state_dict(T.load(self.checkpoint_file))
