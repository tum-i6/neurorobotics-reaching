import os
import torch as T
import torch.nn.functional as F
import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[9],
            action_space=None, gamma=0.99, n_actions=6, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):


        """ Initialize the agent.

            Args:
                alpha: Te learning reate for the actor network
                beta: The learning rate for the value and critic network
                input_dims: Dimension of the obserbation Space
                gamma: The discount factor
                n_actions: Number of action values
                max_size: Max size of the replay buffer
                tau: Factor by which we modulate the parameters of our target value network (we do a soft copy of the value network)
                layer1_size: Size of the first fully connected layer
                layer2_size: Size of the second fully connected layer
                batch_size:
                reward_scale: Accounts for the entropy in the framework - We scale the critic loss by this factor
        """
        # Save the parameters
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Init the networks
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=action_space.high)
        # We use two critic network. We take the minimum of the evaluation of the loss function for the value and actor network
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        # Two value networks that only differ in name
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale

        # Set the init value of the value and target value network to exactly the same
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """ Chooses an action based on the observation 

            Args:
                observation: The perception of the current state

            Return:
                The action proposed by the action model
        """
        # Turn the observation to a tensor and pass it to the device
        state = T.Tensor([observation]).to(self.actor.device)
        # Pass the observation through the action model. Only retrive the action not the loss
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # send the action to the cpu, detach it and turn it in to a numpy array
        return actions.cpu().detach().numpy()[0]


    def remember(self, state, action, reward, new_state, done):
        """ Saves the the transition values to the memory.

            Args:
                state: The initial state
                action: The action of this state
                reward: The reward for this state
                state_: The New state
                done: Termination flag
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """ Performs a soft update of the value network parameters.

            Args:
                tau: Value used to initialize the networks 
        """

        # Make sure that before the first pass (at initialization) value and target value network have exactly the same weights
        if tau is None:
            tau = self.tau

        # Creating a copy of the parameters, modify them and then upload them
        # Get weights from target_value network
        target_value_params = self.target_value.named_parameters()
        # Get weights from value network
        value_params = self.value.named_parameters()

        # Convert the weights to dicts
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        """ Saves the checkpoints for all models
        """
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        """ Loads the checkpoints for all models
        """
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        # Check if we allready filled up at least "batch size" of our memory
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample the buffer
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # Convert the numpy arrays to tensors
        # Important: They all have to be passed to the correct device
        # We can send them to the actor device as all networks use the same
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Calculate the value and target value
        # We have to collapse allonge the batch dimension due to how the loss is calculated later
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        # Every where that the new states are terminal we want to set the value to zero as this is the definition of the value function
        value_[done] = 0.0

        # Get the action and log probability accroding to the new policy
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        # Get the new policy by passing the state and action to the two critic network and keeping the min
        # This is done to combat the overestimation bias and stabilize the learning. (See the TD3 paper)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Calculate the value loss and backpropergate it
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        # Keep PyTorch from discarding the graph after back propergation as there is a coupling between the loss of the different NNs
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Calculate the actor loss and backpropergate it
        # Feedforward - using the reparameterization
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Calculate the critics loss and backpropergate it
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # The scaling vector handels the entropy in our loss function encourage exploration
        q_hat = self.scale*reward + self.gamma*value_
        # We now calculate the old q policy using the data form the replay buffer (action and not actions)
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
