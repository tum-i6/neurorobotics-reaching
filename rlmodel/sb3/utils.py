import matplotlib.pyplot as plt
import numpy as np

from gym import spaces


def scatter_plotter(data, step):
    ''' Creates a scatter plot of the last 100 and all previous samples
    '''
    if (step % 100) == 0:
        # last 100
        data100 = {}
        data100["x"] = data["x"][-100:]
        data100["y"] = data["y"][-100:]
        data100["c"] = data["c"][-100:]


        # show map
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(data100["x"], data100["y"], c=data100["c"])
        axs[0].set_title('last 100 samples')
        axs[1].scatter(data["x"], data["y"], s=0.7, c=data["c"])
        axs[1].set_title('all samples')

        for ax in axs.flat:
            ax.set(xlabel='x-coord', ylabel='y-coord')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()
        
def bound_space(space_low, space_high):
    ''' Create a space were each value is either bounded to the range of 0 to 1, -1 to 0, or -1 to 1.
        If the lower bound of a value is unequal zero the bound is set to -1, if equal zero the bound is set to 0.
        If the upper bound of a value is unequal zero the bound is set to 1, if equal zero the bound is set to 0.
        
        Args:
            space_low: A numpy array containing the lower bounds of the values of the space
            space_high: A numpy array containing the upper bounds of the values of the space
        
        Return:
            The normalized space.
    '''
    # set the observation space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    space_low_mask = np.where(space_low !=0, -1, 0)
    space_high_mask = np.where(space_high !=0, 1, 0)
    observation_space = spaces.Box(low=np.float32(space_low_mask), 
                          high=np.float32(space_high_mask))    
    return observation_space

def bound_space_her(obs_space_low, obs_space_high, desired_goal_low, desired_goal_high, achieved_goal_low, achieved_goal_high):
    ''' Create a space were each value is either bounded to the range of 0 to 1, -1 to 0, or -1 to 1.
        If the lower bound of a value is unequal zero the bound is set to -1, if equal zero the bound is set to 0.
        If the upper bound of a value is unequal zero the bound is set to 1, if equal zero the bound is set to 0.
        
        Args:
            obs_space_low: A numpy array containing the lower bounds of the values of the observation space
            obs_space_high: A numpy array containing the upper bounds of the values of the observation space
            desired_goal_low: A numpy array containing the lower bounds of the values of the desired goal space
            desired_goal_high: A numpy array containing the upper bounds of the values of the desired goal space
            achieved_goal_low: A numpy array containing the lower bounds of the values of the achieved goal space
            achieved_goal_high: A numpy array containing the lower bounds of the values of the achived goal space 
        
        Return:
            The normalized space.
    '''
    # set the desired goal space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    desired_goal_space_low_mask = np.where(desired_goal_low !=0, -1, 0)
    desired_goal_space_high_mask = np.where(desired_goal_high !=0, 1, 0)
    
    # set the achieved goal space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    achieved_goal_space_low_mask = np.where(achieved_goal_low !=0, -1, 0)
    achieved_goal_space_high_mask = np.where(achieved_goal_high !=0, 1, 0)
    
    # set the observation space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    obs_space_low_mask = np.where(obs_space_low !=0, -1, 0)
    obs_space_high_mask = np.where(obs_space_high !=0, 1, 0)
    
    observation_space = spaces.Dict(dict(
            desired_goal = spaces.Box(low=np.float32(desired_goal_space_low_mask), high=np.float32(desired_goal_space_high_mask)),
            achieved_goal = spaces.Box(low=np.float32(achieved_goal_space_low_mask), high=np.float32(achieved_goal_space_high_mask)),
            observation = spaces.Box(low=np.float32(obs_space_low_mask),high=np.float32(obs_space_high_mask))
            ))
    return observation_space

def bound_space_her_image(obs_space_low, obs_space_high, desired_goal_low, desired_goal_high, achieved_goal_low, achieved_goal_high, width, height, channel):
    ''' Create a space were each value is either bounded to the range of 0 to 1, -1 to 0, or -1 to 1. (only used in image+her)
        If the lower bound of a value is unequal zero the bound is set to -1, if equal zero the bound is set to 0.
        If the upper bound of a value is unequal zero the bound is set to 1, if equal zero the bound is set to 0.
        
        Args:
            obs_space_low: A numpy array containing the lower bounds of the values of the observation space
            obs_space_high: A numpy array containing the upper bounds of the values of the observation space
            desired_goal_low: A numpy array containing the lower bounds of the values of the desired goal space
            desired_goal_high: A numpy array containing the upper bounds of the values of the desired goal space
            achieved_goal_low: A numpy array containing the lower bounds of the values of the achieved goal space
            achieved_goal_high: A numpy array containing the lower bounds of the values of the achived goal space 
            width: Int, width of image
            height: Int, height of image
            channel: Int, channel of image
        
        Return:
            The normalized space.
    '''
    # set the desired goal space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    desired_goal_space_low_mask = np.where(desired_goal_low !=0, -1, 0)
    desired_goal_space_high_mask = np.where(desired_goal_high !=0, 1, 0)
    
    # set the achieved goal space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    achieved_goal_space_low_mask = np.where(achieved_goal_low !=0, -1, 0)
    achieved_goal_space_high_mask = np.where(achieved_goal_high !=0, 1, 0)
    
    # set the observation space values to the range of -1 to 1, -1 to 0, or 0 to 1 depending on the actual range
    obs_space_low_mask = np.where(obs_space_low !=0, -1, 0)
    obs_space_high_mask = np.where(obs_space_high !=0, 1, 0)
    
    observation_space = spaces.Dict(dict(
            desired_goal = spaces.Box(low=np.float32(desired_goal_space_low_mask), high=np.float32(desired_goal_space_high_mask)),
            achieved_goal = spaces.Box(low=np.float32(achieved_goal_space_low_mask), high=np.float32(achieved_goal_space_high_mask)),
            observation = spaces.Dict(
                    spaces = {
                        "vec": spaces.Box(low=np.float32(obs_space_low_mask), high=np.float32(obs_space_high_mask)),
                        "img": spaces.Box(low=0, high=255,shape=(width, height, channel), 
                                          dtype=np.uint8),
                    }
                )
    )
    )
    return observation_space

def normalize_space(space, space_low, space_high, space_low_mask, space_high_mask):
    ''' Normalizes a space input to the expected -1 to 1, -1 to 0, or 0 to 1 range bounds.
    
        Args:
            space: The space representation (the observation)
            space_low: The low bounds of the space
            space_high: The upper bounds of the space
            space_low_mask: The lower mask of the space (-1 or 0 values)
            space_high_mask: The upper mask of the space (1 or 0 values)
    
        Return:
            A normalized version of the space input in accordance with the boundaries set by bound_space().    
    '''
    for i, s in enumerate(space):
        # if the lower and uper bound are zero the observation should also be zero
        if space_low_mask[i] == 0 and space_high_mask[i] == 0:
            space[i] = 0
        # if the lower bound is zero normalize between 0 and 1
        elif space_low_mask[i] == 0:
            space[i] = (s - space_low[i]) / (space_high[i] - space_low[i])
        # if the upper bound is zero normalize between -1 and 0
        elif space_high_mask[i] == 0:
            space[i] = (s - space_low[i]) / (space_high[i] - space_low[i]) - 1
        # if lower and upper bound are unequal zero normalize between -1 and 1
        else:
            space[i] = ((s - space_low[i]) / (space_high[i] - space_low[i]) - 0.5) * 2
    return space
    
def de_normalize_space(space, space_low, space_high, space_low_mask, space_high_mask):
    ''' De-normalizes a space input to the original range.
    
        Args:
            space: The space representation (the action space)
            space_low: The low bounds of the space
            space_high: The upper bounds of the space
            space_low_mask: The lower mask of the space (-1 or 0 values)
            space_high_mask: The upper mask of the space (1 or 0 values)
    
        Return:
            The de-normalized version of the space input in accordance with the boundaries set by bound_space().    
    '''
    # reverse the normalization of the action space 
    for i, s in enumerate(space):
        # if the lower and uper bound are zero the observation should also be zero
        if  space_low_mask[i] == 0 and space_high_mask[i] == 0:
            space[i] = 0
        # if the lower bound is zero normalize between 0 and 1
        elif space_low_mask[i] == 0:
            space[i] = s * (space_high[i] - space_low[i]) + space_low[i]
        # if the upper bound is zero normalize between -1 and 0
        elif space_high_mask[i] == 0:
            space[i] = (s + 1)* (space_high[i] - space_low[i]) + space_low[i]
        # if lower and upper bound are unequal zero normalize between -1 and 1
        else:
            space[i] = (s / 2 + 0.5)* (space_high[i] - space_low[i]) + space_low[i]
    return space

