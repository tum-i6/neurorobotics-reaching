## Soft Actor Critic 

Custom SAC implementation whose logic is divided into the following three scripts:
* [buffer.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/custom_models/sac/buffer.py)
* [networks.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/custom_models/sac/networks.py])
* [sac_torch.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/custom_models/sac/sac_torch.py)

There are two main implementations - main files:
* A basic custom environment implementation [main_sac_nrp.py](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/custom_models/sac/main_sac_nrp.py)
* [Custom_SAC.ipynb](https://gitlab.lrz.de/cmlr_ss_21/G2_ReachingTask/-/tree/master/rlmodel/custom_models/sac/SAC.ipynb) utilising the SB3 environment, located in the parent folder (`../`).

### Buffer

The file `buffer.py` holds the implementation of the replay buffer. It initializes the memory using numpy arrays. At start, the replay buffer is filled up with the initial state, the action, the reward, the final state, and a done flag for each episode.  Later the model can replay these saved transitions.

### Networks
The file `networks.py` specifies the critic, value, and actor network.

##### Critic Network
The critic network takes a state and an action as input and tells the actor network if the chosen action was good or bad. Following the original SAC paper, the critic network is a basic FCN with three layers and Relu activation functions.

##### Value Network
The value network evaluates if a state is valuable or not - it does not care about the action you take or took.
Therefore, the network has the state as single input. Otherwise, it is just like the critic network, a simply three-layered FCN with Relu activation functions.

##### Actor Network
The actor network handles how the agent selects an action.  The network outputs a mean and standard deviation. 
The network has two consecutive FC layers with Relu activations before splitting into two heads. Both heads are comprised of a single FC layer. One head calculates the mean, and the other calculates the standard deviation.
The standard deviation is then clamped to limit the spread of the distribution. For the output, the network then samples from the normal distribution.

### Agent

The file `sac_torch.py` implements the SAC agent. Given an observation, the agent chooses an action using the action network.  After executing an action, the agent fills the replay buffer with the initial observation, the chosen action, the calculated return, and the final observation. When the replay buffer is full, the agents start the learning process. In the learning step, the agent calculates the value of the given state, updates the policy using the critic network, and updated the network parameters.




