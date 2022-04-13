import numpy as np
import math


def augment_cylinder_position(cylinder_pos, pars):
    ''' Returns a random position on the table.

        Args:
            cylinder_pos: if 'pars["CYLINDER"]' is 'no' this will be the cylinder position
            pars: position space depends on 'pars["CYLINDER"]' argument
                'fix': fix position at [0.45, 0.40, 1.12]
                'semi_random': see rand_bow()
                'semi_random_sides': see rand_bow()
                'half_table': see rand_half_table()
                '3/4-table': see rand_three_quarte_table()
                '7/8-table': see rand_seven_eight_table()
                'whole_table': see rand_whole_table()
                'no': no augmentation -> just makes np.array from coordinates

    '''
    if pars["CYLINDER"] == 'fix':
        print("  [reset]: fix pos: [0.45, 0.40, 1.12]")
        return [0.45, 0.40, 1.12]
    elif pars["CYLINDER"] == 'semi_random':
        pos_cyl = rand_bow()
        print("  [reset]: semi-random pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == 'semi_random_sides':
        pos_cyl = rand_bow(prefer_sides=True)
        print("  [reset]: semi-random-sides pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == 'half_table':
        pos_cyl = rand_half_table()
        print("  [reset]: half-table pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == '3/4-table':
        pos_cyl = rand_three_quarter_table()
        print("  [reset]: 3/4-table pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == '7/8-table':
        pos_cyl = rand_seven_eight_table()
        print("  [reset]: 7/8-table pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == 'whole_table':
        pos_cyl = rand_whole_table()
        print("  [reset]: whole_table pos: ", pos_cyl)
        return pos_cyl
    elif pars["CYLINDER"] == 'no':
        x = cylinder_pos[0]
        y = cylinder_pos[1]
        z = 1.12
        pos_cyl = np.array([x, y, z])
        print("  [reset]: position: ", pos_cyl)
        return pos_cyl

def rand_bow(prefer_sides=False):
    """ Returns a (reachable) random cylinder position for the 'reduced' robot setting on a bow.

        Args:
            prefer_sides: used to sample only on the sides of the bow
    """
    # sample from -0.44 to 0.48
    x = np.random.uniform(low=-0.44, high=0.48)
    if prefer_sides:
        while (x > -0.34) and (x < 0.38):
            x=np.random.uniform(low=-0.44, high=0.48)
        
    # function for bow
    y = 2.3 * (x-0.02) * (x-0.02) - 0.05
    
    z = 1.12
    
    return np.array([x, y, z])

def rand_half_table():
    """ Returns a (reachable) random cylinder position for the 'reduced' robot setting on the nearer half of the table
    """
    # sample x from -0.44 to 0.48
    x = np.random.uniform(low=-0.44, high=0.48)

    # sample y from 0.0 to 0.44
    y = np.random.uniform(low=0.0, high=0.44)
    
    # z is fix
    z = 1.12
    
    # correct points too near/far the robot
    radius = math.sqrt(x**2 + (y-0.4)**2)
    while (radius < 0.2) or (radius > 0.46):
        # resample
        x = np.random.uniform(low=-0.44, high=0.48)
        y = np.random.uniform(low=0.0, high=0.44)
        
        # new radius
        radius = math.sqrt(x**2 + (y-0.4)**2)

    return np.array([x, y, z])

def rand_three_quarter_table():
    """ Returns a (reachable) random cylinder position for the 'reduced' robot setting on the nearer three-quarter of the table
    """
    # sample x from -0.44 to 0.48
    x = np.random.uniform(low=-0.44, high=0.48)

    # sample y from -0.24 to 0.44
    y = np.random.uniform(low=-0.24, high=0.44)
    
    # z is fix
    z = 1.12
    
    # correct points too near/far the robot
    radius = math.sqrt(x**2 + (y-0.4)**2)
    while (radius < 0.2) or (radius > 0.66):
        # resample
        x = np.random.uniform(low=-0.44, high=0.48)
        y = np.random.uniform(low=-0.24, high=0.44)
        
        # new radius
        radius = math.sqrt(x**2 + (y-0.4)**2)

    return np.array([x, y, z])
      
def rand_seven_eight_table():
    """ Returns a (reachable) random cylinder position for the 'reduced' robot setting on the nearer seven-eight of the table
    """
    # sample x from -0.44 to 0.48
    x = np.random.uniform(low=-0.44, high=0.48)

    # sample y from -0.34 to 0.44
    y = np.random.uniform(low=-0.34, high=0.44)
    
    # z is fix
    z = 1.12
    
    # correct points too near/far the robot
    radius = math.sqrt(x**2 + (y-0.4)**2)
    while (radius < 0.2) or (radius > 0.78):
        # resample
        x = np.random.uniform(low=-0.44, high=0.48)
        y = np.random.uniform(low=-0.34, high=0.0) # to force more samples at upper part
        
        # new radius
        radius = math.sqrt(x**2 + (y-0.4)**2)

    return np.array([x, y, z])

def rand_whole_table():
    """ Returns a (reachable) random cylinder position for the 'reduced' robot setting on the table
    """
    # sample x from -0.44 to 0.48
    x = np.random.uniform(low=-0.44, high=0.48)

    # sample y from -0.34 to 0.44
    y = np.random.uniform(low=-0.48, high=0.44)
    
    # z is fix
    z = 1.12
    
    # correct points too near/far the robot
    radius = math.sqrt(x**2 + (y-0.4)**2)
    while (radius < 0.2):
        # resample
        x = np.random.uniform(low=-0.44, high=0.48)
        y = np.random.uniform(low=-0.48, high=0.44) 
        
        # new radius
        radius = math.sqrt(x**2 + (y-0.4)**2)

    return np.array([x, y, z])