# my_project/config.py
import numpy as np
from yacs.config import CfgNode as CN

_C = CN()

###### AGENT ######

_C.AGENT = CN()

# Alpha value
_C.AGENT.ALPHA = 0.0003

# Beta value
_C.AGENT.BETA = 0.0003

# Gamma value
_C.AGENT.GAMMA = 0.99

# Max memory size
_C.AGENT.MAX_SIZE = 1000000

# Tau Value
_C.AGENT.TAU = 0.005

# Size of the first ff connected layer
_C.AGENT.LAYER_ONE = 256

# Size of the second ff connected layer
_C.AGENT.LAYER_TWO = 256

# Batch size for the replay buffer
_C.AGENT.BATCH_SIZE = 256

# Hyper parameter scaling exploration vs exploitation
_C.AGENT.REWARD_SCALE = 2.0

###### SPACES ######

_C.SPACES = CN()

# Min x value for the cylinder postion
_C.SPACES.CYLINDER_X_LOW = -1.56

# Min y value for the cylinder postion
_C.SPACES.CYLINDER_Y_LOW = -1.60

# Min z value for the cylinder postion
_C.SPACES.CYLINDER_Z_LOW = 0.12

# Max x value for the cylinder postion
_C.SPACES.CYLINDER_X_HIGH = 1.60

# Max y value for the cylinder postion
_C.SPACES.CYLINDER_Y_HIGH = 1.56

# Max z value for the cylinder postion
_C.SPACES.CYLINDER_Z_HIGH = 1.12

# Min angle for the robot joints
_C.SPACES.ROB_ANGLE_LOW = -np.pi

# Max angle for the robot joints
_C.SPACES.ROB_ANGLE_HIGH = np.pi

###### TRAIN ######

_C.TRAIN = CN()

# Max number of games
_C.TRAIN.NR_GAMES = 100000

# Max value of the reward function
_C.TRAIN.MAX_REWARD = 15.0

# Min value of the reward function
_C.TRAIN.MIN_REWARD = -1.0

# Boolean stating if a the last checkpoint should be loaded
_C.TRAIN.LOAD_CHECKPOINT = False

# Boolean stating that only on run per episode should be taken
_C.TRAIN.SINGLE_ACTION_PER_ROUND = True

# Max number of runs per episode
_C.TRAIN.MAX_ROUNDS_PER_EPISODE = True

# Trashhold for proximity to cylinder that rewards the max reward
_C.TRAIN.THRESHOLD = 0.15



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`