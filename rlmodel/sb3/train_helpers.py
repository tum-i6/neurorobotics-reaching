import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from gym import spaces


def get_action_space(pars):
    ''' Returns an action space with different numbers of actuated joints.
        
        Args:
            pars: Number of joints depends on 'pars["SETTING"]' argument
                'reduced': 1st joint
                'reduced2': 1st, 5th joint
                'reduced3': 1st, 2nd, 3rd joint
                'reduced3+': 1st, 3rd, 5th joint (bigger space)
                'reduced4': 1st, 2nd, 3rd, 5th joint
                'reduced4+': 1st, 2nd, 3rd, 5th joint (bigger space)
    '''
    if pars["SETTING"] == 'reduced':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -0.01, np.pi/2.01, -0.01, np.pi/3.01, -0.01])), 
                          high=np.float32(np.array([np.pi/2, 0.01, np.pi/1.99, 0.01, np.pi/2.99, 0.01])))    
    elif pars["SETTING"] == 'reduced2':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -0.01, np.pi/2.01, -0.01, 0, -0.01])), 
                          high=np.float32(np.array([np.pi/2, 0.01, np.pi/1.99, 0.01, np.pi/2, 0.01])))    
    elif pars["SETTING"] == 'reduced3':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -np.pi/2, 0, -0.001, -0.001, -0.001])), 
                          high=np.float32(np.array([np.pi/2, 0, np.pi, 0.001, 0.001, 0.001])))  
    elif pars["SETTING"] == 'reduced3+':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -0.01, (np.pi/2)-(np.pi/4), -0.01, 0, -0.01])), 
                          high=np.float32(np.array([np.pi/2, 0.01, (np.pi/2)+(np.pi/4), 0.01, np.pi/2, 0.01])))    
    elif pars["SETTING"] == 'reduced4':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -np.pi/3, (np.pi/2)-(np.pi/4), -0.01, 0, -0.01])), 
                          high=np.float32(np.array([np.pi/2, 0.0, (np.pi/2)+(np.pi/4), 0.01, np.pi/2, 0.01])))    
    elif pars["SETTING"] == 'reduced4+':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -(2*np.pi)/3, 0, -0.01, 0, -0.01])), 
                          high=np.float32(np.array([np.pi/2, 0.0, (np.pi/2)+(np.pi/4), 0.01, np.pi/2, 0.01])))
    elif pars["SETTING"] == 'full_constrained':
        return spaces.Box(low=np.float32(np.array([-np.pi/2, -2*np.pi/3, 0, -np.pi/2, 0, -np.pi])), 
                          high=np.float32(np.array([np.pi/2, 0.0, 3*np.pi/4, np.pi/2, np.pi/2, np.pi])))
    elif pars["SETTING"] == 'full':
        return spaces.Box(low=np.float32(np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])), 
                          high=np.float32(np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])))                       

def get_reward(dist, threshold, pars):
    ''' Returns the reward.

        Args: 
            dist: the distance to the cylinder
            threshold: the current valid threshold
            pars: the specified reward type pars["REWARD_TYPE"]
    '''
    reward = 0.0
    if pars["REWARD_TYPE"] == 'sparse':
        # (sparse) reward
        if dist < threshold:
            reward = 1.0
        else:
            reward = 0.0
    elif pars["REWARD_TYPE"] == 'dense':
        # (dense) reward
        if dist < threshold:
            reward = 1.0
        else:
            reward = -dist
    elif pars["REWARD_TYPE"] == 'extra_dense':
        # (extra dense) reward
        if dist < (threshold/2):
            reward = 3.0
        elif dist < threshold:
            reward = 1.0
        else:
            reward = -dist
    else:
        print("  [step]: reward type not specified")
    
    if pars["TOGGLE_REWARD"]:
        reward = -reward
    
    print("  [step]: reward: ", reward)
    return reward
    
def threshold_scheduling(history, cur_threshold, pars, vel=15):
    ''' Reduces the threshold based on the current performance.

        Args:
            history: Array consisting of ones (if episode succesfull) and zeros (episode unsuccesfull)
            cur_threshold: Current distance threshold. The agent should at least reach this proximity to the target to consider an episode succesfull
            pars: Dictionary of global parameters. Threshold value should not go below pars["MIN_THRESHOLD"] 
            vel: "Velocity" of threshold reduction. Number of succesfull episodes needed for threshold reduction.

        Returns:
            threshold: New threshold value
            history: history after threshold reduction (appended 0 to avoid threshold reduction in unwanted consecutive steps)
    '''
    # if last 15 steps were successful, than reduces threshold by 2cm
    threshold = cur_threshold
    
    if len(history) >= vel:
        if all(y == 1 for y in history[-vel:]):
            if (cur_threshold > pars["MIN_THRESHOLD"]) and (cur_threshold < 0.11):
                history.append(0)
                threshold -= 0.01
                print("[threshold_scheduling]: threshold changed to ", threshold)
            elif cur_threshold > pars["MIN_THRESHOLD"]:
                history.append(0)
                threshold -= 0.02
                print("[threshold_scheduling]: threshold changed to ", threshold)
    
    return threshold, history

def evaluate(model, env, params, writer=None):
    """ Run several episodes to show avg distance
        
        Args:
            model (agent): agent to evaluate
            env (env): gym environment
            params: dict containing info
            writer: instance of SummaryWriter (tensorboard logging)
        
        Returns:
            avg_distance (float): average distance to goal
            avg_reward (float): average reward over the number of evaluations
            success_rate (%): percentage of successful episodes
    """
    print("Evaluation")
    data = {}
        
    for e in (params["EVALS"]):
        # collect points for map
        print("e: ", e)
        data[e] = {"x" : [0, 0], 
                   "y" : [0, 0],
                   "c" : [0, 0],
                   "avg_distance": 0,
                   "success_rate": 0}
        x_list = []
        y_list = []
        c_list = []

        obs = env.reset()
        done_eval_steps = 0
        successful = 0
        failed = 0
        while done_eval_steps < params["EVALUATION_STEPS"]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            x, y, _ = info['cyl position']
            x_list.append(x)
            y_list.append(y)
            data[e]["avg_distance"] += info['total_distance']
            if info['total_distance'] < e:
                successful += 1
                c_list.append("green")
            else:
                failed += 1
                c_list.append("red")
            obs = env.reset()
            done_eval_steps += 1

        # add data
        data[e]["x"] = x_list
        data[e]["y"] = y_list
        data[e]["c"] = c_list
        data[e]["avg_distance"] /= params["EVALUATION_STEPS"]
        data[e]["success_rate"] = successful/params["EVALUATION_STEPS"]

        # success rate:
        print("eval distance: ", e)
        print("Avg distance: ", data[e]["avg_distance"]) 
        print("Success rate: ", data[e]["success_rate"])     
            
        # show map
        fig, ax = plt.subplots()
        ax.scatter(data[e]["x"], data[e]["y"], c=data[e]["c"])
        ax.set_title("Threshold:{:.2f}m Success rate:{:.3f} Avg distance:{:.3f}m".format(e, data[e]["success_rate"], data[e]["avg_distance"]))
        ax.set(xlabel='x-coord [m]', ylabel='y-coord [m]')
        ax.set_xlim((-0.44,0.48))
        ax.set_ylim((-0.48,0.44))
        plt.show()

        if writer is not None:
            writer.add_figure(f"Model performance evaluation (threshold: {e}m)", fig)
            
def evaluate2(model, env, params, writer=None):
    """ Run several episodes to show avg distance
        -> this function uses the same points for evaluation of different thresholds
        
        Args:
            model (agent): agent to evaluate
            env (env): gym environment
            params: dict containing info
            writer: instance of SummaryWriter (tensorboard logging)
        
        Returns:
            avg_distance (float): average distance to goal
            avg_reward (float): average reward over the number of evaluations
            success_rate (%): percentage of successful episodes
    """
    print("Evaluation")
    data = {}
        
    x_list = []
    y_list = []
    d_list = []
    avg_distance = 0

    obs = env.reset()
    done_eval_steps = 0
    while done_eval_steps < params["EVALUATION_STEPS"]:
        print("eval_step: ", done_eval_steps+1)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        x, y, _ = info['cyl position']
        x_list.append(x)
        y_list.append(y)
        d_list.append(info['total_distance'])
        avg_distance += info['total_distance']
        obs = env.reset()
        done_eval_steps += 1
           
    for e in (params["EVALS"]):
        # specify data
        data[e] = {"x" : [0, 0], 
                   "y" : [0, 0],
                   "c" : [0, 0],
                   "d" : 0,
                   "avg_distance": 0,
                   "success_rate": 0}
        # add data
        data[e]["x"] = x_list
        data[e]["y"] = y_list
        data[e]["d"] = d_list
        data[e]["avg_distance"] = avg_distance/params["EVALUATION_STEPS"]
        successful = 0
        failed = 0

        c_list = []
        for d in d_list:
            if d < e:
                successful += 1
                c_list.append("green")
            else:
                failed += 1
                c_list.append("red")
        
        data[e]["c"] = c_list
        data[e]["success_rate"] = successful/params["EVALUATION_STEPS"]
        
        # success rate:
        print("eval distance: ", e)
        print("Avg distance: ", data[e]["avg_distance"]) 
        print("Success rate: ", data[e]["success_rate"])     
            
        # show map
        fig, ax = plt.subplots()
                
        ax.scatter(data[e]["x"], data[e]["y"], c=data[e]["c"])
        ax.set_title("Threshold:{:.2f}m Success rate:{:.3f} Avg distance:{:.3f}m".format(e, data[e]["success_rate"], data[e]["avg_distance"]))
        ax.set(xlabel='x-coord [m]', ylabel='y-coord [m]')
        ax.set_xlim((-0.44,0.48))
        ax.set_ylim((-0.48,0.44))
                
        plt.show()

        if writer is not None:
            writer.add_figure(f"Model performance evaluation (threshold: {e}m)", fig)
            

def evaluate3(model, env, params, writer=None, max_threshold=0.1):
    """ Run several episodes to show avg distance
        -> this function uses the same points for evaluation of different thresholds
        -> this function color encodes the distance for each attempt
        
        Args:
            model (agent): agent to evaluate
            env (env): gym environment
            params: dict containing info
            writer: instance of SummaryWriter (tensorboard logging)
            max_treshold: color encoding goes from 0.0 to max_treshold
        
        Returns:
            avg_distance (float): average distance to goal
            avg_reward (float): average reward over the number of evaluations
            success_rate (%): percentage of successful episodes
    """
    print("Evaluation")
    data = {}
        
    x_list = []
    y_list = []
    d_list = []
    avg_distance = 0

    obs = env.reset()
    done_eval_steps = 0
    while done_eval_steps < params["EVALUATION_STEPS"]:
        print("eval_step: ", done_eval_steps+1)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        x, y, _ = info['cyl position']
        x_list.append(x)
        y_list.append(y)
        d_list.append(info['total_distance'])
        avg_distance += info['total_distance']
        obs = env.reset()
        done_eval_steps += 1
           
    for e in (params["EVALS"]):
        # specify data
        data[e] = {"x" : [0, 0], 
                   "y" : [0, 0],
                   "c" : [0, 0],
                   "d" : 0,
                   "avg_distance": 0,
                   "success_rate": 0}
        # add data
        data[e]["x"] = x_list
        data[e]["y"] = y_list
        data[e]["d"] = d_list
        data[e]["avg_distance"] = avg_distance/params["EVALUATION_STEPS"]
        successful = 0
        failed = 0
        
        norm = mpl.colors.Normalize(vmin=0.0, vmax=max_threshold)
        cmap = cm.RdYlGn_r
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        c_list = []
        for d in d_list:
            c_list.append(m.to_rgba(d))
            if d < e:
                successful += 1
            else:
                failed += 1
        
        data[e]["c"] = c_list
        data[e]["success_rate"] = successful/params["EVALUATION_STEPS"]
        
        # success rate:
        print("eval distance: ", e)
        print("Avg distance: ", data[e]["avg_distance"]) 
        print("Success rate: ", data[e]["success_rate"])     
            
    # show map
    fig, ax = plt.subplots()

    evals = params["EVALS"]
    ax.scatter(data[evals[0]]["x"], data[evals[0]]["y"], c=data[evals[0]]["c"])
    ax.set_title("Threshold: {:.2f}m   Success rate: {:.1f}% \n Threshold: {:.2f}m   Success rate: {:.1f}% \n Threshold: {:.2f}m   Success rate: {:.1f}% \n Threshold: {:.2f}m   Success rate: {:.1f}% \n Avg distance: {:.3f}m".format( 
                evals[0], data[evals[0]]["success_rate"]*100, evals[1] , data[evals[1]]["success_rate"]*100, evals[2] , data[evals[2]]["success_rate"]*100, evals[3] , data[evals[3]]["success_rate"]*100, data[evals[0]]["avg_distance"]))
    ax.set(xlabel='x-coord [m]', ylabel='y-coord [m]')
    ax.set_xlim((-0.44,0.48))
    ax.set_ylim((-0.48,0.44))

    fig.colorbar(m, orientation ='vertical')

    plt.show()

    if writer is not None:
        writer.add_figure(f"Model performance evaluation", fig)

def train(agent, env, params):
    """ Train the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run
    
    """
    
    done_episodes = 0
    
    while done_episodes < params["EXPLORATION"]:
        agent = agent.learn(total_timesteps=params["STEPS"], log_interval=1)
        done_episodes += 1
