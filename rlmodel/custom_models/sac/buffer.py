import numpy as np

# The memory of our agent

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        """ Initialize the agents memory buffer

            Args:
                max_size: maximum size of the memory (we do not want it to be unbounded)
                input_shape: observation dimensionality of the environment
                n_actions: number of components of the action (since continour env.)
        """
        self.mem_size = max_size
        # Memory counter - keeps track of postion of the first available memory
        self.mem_cntr = 0
        # Init memory
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        # State memory that keeps track of the states that result after the actions
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        # Saves the action values for each memory step
        self.action_memory = np.zeros((self.mem_size, n_actions))
        # Keep track the rewards the agent recives
        self.reward_memory = np.zeros(self.mem_size)
        # Saves the done flags to set the values of the terminal states to zero ??? 
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """ Stores the transition information

            Args:
                state: The initial state
                action: The action of this state
                reward: The reward for this state
                state_: The New state
                done: Termination flag
        """

        # Figure out where the first free memory is
        index = self.mem_cntr % self.mem_size

        # The the transition information to the relevant memory array
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Increment the memory counter
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """ Used to sample the buffer

            Args:
                batch_size: The batch size
                
            Returns:
                states: The initale state (observation)
                actions: The choosen actions
                rewards: The given rewards
                states_: The states after the action was executed
                dones: The done flags
        """

        # Number of memories saved in the buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        # Define the batch size as either the max_mem or the batch_size
        batch = np.random.choice(max_mem, batch_size)

        # Sample the memories
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

