import sys
import numpy as np

from gym import spaces
from utils import scatter_plotter

class Environment(object):
    """ Wrapper class having the same method structure as the environments in gym.
    
        Args:
            experiment (GRPCClient): client to communicate via gRPC with the simulation backend (to call experiment_api)  
            threshold (float): Distance to the goal the robot agent should reach
            random_goal (bool): If True, spawn the cylinder randomly in each episode. If False, the target position is fixed throughout the training.
            use_3_joints (bool): If True, the robot should use only the first 3 joints to try to reach the target. If False, use all 6 joints.            
    """
    def __init__(self, experiment, threshold = 0.25, random_goal=True, use_3_joints=False):
        self.experiment = experiment
        self.threshold = threshold
        self.pos_cyl = self.experiment.setup()
        self.r_js = []
        self.pos_ee = []
        self.dist_ee_cyl = None
        self.random_goal = random_goal
        self.use_3_joints = use_3_joints
        
        # Observation space: relative position ee-cyl (x, y, z), joint angles    
        self.observation_space = spaces.Box(low=np.float32(np.array([-1.5, -1, -0.55, -np.pi/2, -np.pi/2, 0, -np.pi/2, 0, -np.pi])),
                                            high=np.float32(np.array([1.5, 1.8, 1, np.pi/2, 0, np.pi, np.pi/2, np.pi, np.pi])))
        # self.observation_space = spaces.Box(low=np.float32(np.array([-1.5, -1, -0.55, -0.44, -0.48, 1.12, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])),
        #                                     high=np.float32(np.array([1.5, 1.8, 1, 0.48, 0.44, 1.12, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])))

        # Action space of the robot joint_1, joint_2, joint_3, joint_4, joint_5
        self.action_space = spaces.Box(low=np.float32(np.array([-np.pi/2, -np.pi/2, 0, -np.pi/2, 0, -np.pi])), 
                                       high=np.float32(np.array([np.pi/2, 0, np.pi, np.pi/2, np.pi, np.pi])))

        if self.use_3_joints:
            # Action space of the robot joint_1, joint_2, joint_3
            self.action_space = spaces.Box(low=np.float32(np.array([-np.pi/2, -np.pi/2, 0, 0, 0, 0])), 
                                           high=np.float32(np.array([np.pi/2, 0, np.pi, 0, 0, 0])))         
        
               
    def reset(self):
        """ Resets the environment and returns the current observation of the state.
        """
               
        if self.random_goal:        
            self.pos_cyl = self.experiment.setup()
        else:
            self.experiment.setup()

        self.r_js = self.experiment.robot.get_joint_states()
        pos_joint1, _, _, self.pos_ee = self.experiment.robot.get_position()
        rel_pos_ee_cyl = np.asarray(self.pos_ee,dtype=np.float32) - np.asarray(self.pos_cyl,dtype=np.float32)
        rel_pos_ee_joint1 = np.asarray(self.pos_ee,dtype=np.float32) - np.asarray(pos_joint1,dtype=np.float32)
        self.dist_ee_cyl = np.linalg.norm(rel_pos_ee_cyl)
        
        # Choose some variant of observation (choose observation space accordingly)
        obs = [*rel_pos_ee_cyl, *self.r_js]
        # obs = [*rel_pos_ee_cyl, *self.pos_cyl, *self.r_js]
        # obs = [*rel_pos_ee_joint1, *rel_pos_ee_cyl, *self.pos_cyl, *self.r_js]
        return obs
        
    def step(self, action):
        """ Executes the input action in the environment and returns the new observation, the reward for the action, 
            whether the episode has terminated and additional information.
        """

        previous_distance = self.dist_ee_cyl
        self.experiment.execute(action)
        self.r_js = self.experiment.robot.get_joint_states()
        pos_joint1, _, _, self.pos_ee = self.experiment.robot.get_position()
        rel_pos_ee_cyl = np.asarray(self.pos_ee,dtype=np.float32) - np.asarray(self.pos_cyl,dtype=np.float32)
        rel_pos_ee_joint1 = np.asarray(self.pos_ee,dtype=np.float32) - np.asarray(pos_joint1,dtype=np.float32)
        self.dist_ee_cyl = np.linalg.norm(rel_pos_ee_cyl)
        
        # Choose some variant of observation (must be same as in reset())
        obs = [*rel_pos_ee_cyl, *self.r_js]
        # obs = [*rel_pos_ee_cyl, *self.pos_cyl, *self.r_js]
        # obs = [*rel_pos_ee_joint1, *rel_pos_ee_cyl, *self.pos_cyl, *self.r_js]
        
        # Choose the reward function you would like to use
        # reward = (np.abs(previous_distance - self.dist_ee_cyl)) / self.dist_ee_cyl
        # reward = 1 / self.dist_ee_cyl
        if (self.dist_ee_cyl < self.threshold): # sparse + dense reward
            reward = 1
        else:
            reward = -self.dist_ee_cyl
        # if (self.dist_ee_cyl < self.threshold): # sparse + dense reward
        #     reward = previous_distance - self.dist_ee_cyl + 10
        # else:
        #     reward = previous_distance - self.dist_ee_cyl
    
        # punishment for colliding with the table
        # if rel_pos_ee_cyl[2] < 0: 
        #     reward -= 2
        
        if (self.dist_ee_cyl < self.threshold): # close enough to the target
            done = True
        else:
            done = False
                            
        info = {
            "distance": self.dist_ee_cyl,
            "pos_cyl": self.pos_cyl} 
        
        return obs, reward, done, info


class Runner():
    """ Carries out the environment steps and adds experiences to memory. This is used in the training process of the agent.
    
        Args:
            env (Environment): gym environment
            agent (TD3): agent to train and evaluate
            replay_buffer (ReplayBuffer): buffer to collect experience  
    """
    
    def __init__(self, env, agent, replay_buffer):
        
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False
        
    def next_step(self, episode_timesteps, noise=0.1):
        
        action = self.agent.select_action(np.array(self.obs), expl_noise=noise)
        
        # Perform action
        new_obs, reward, done, info = self.env.step(action) 
        done_bool = 0 if episode_timesteps + 1 == 200 else float(done)
    
        # Store data in replay buffer
        self.replay_buffer.store(self.obs, action, new_obs, reward, done_bool)
        
        self.obs = new_obs
        
        if done:
            self.obs = self.env.reset()
            done = False            
            return reward, True, info["distance"]
        
        return reward, done, info["distance"]

def evaluate_policy(policy, env, eval_episodes=10, tries=10, plot_scatter=False):
    """ Run several episodes using the best agent policy
        
        Args:
            policy (TD3): agent to train and evaluate
            env (Environment): gym environment
            eval_episodes (int): how many test episodes to run
            tries (int): how many tries does the robot have in an episode
            plot_scatter (bool): whether to create scatter plot using the collected data
        
        Returns:
            avg_reward (float): average reward over the number of evaluations
            avg_success (float): average number of episodes in which the agent succeded in solving the task
            avg_distance (float): average distance to the goal over the number of evaluations
            fig (matplotlib.pyplot.figure): if plot_scatter is True, return a scatter plot containing the result of each episode
    
    """
    avg_success = 0.
    avg_reward = 0.
    avg_distance = 0.
    data_scatter = {"x": [], "y": [], "c": []}
    pos_cyl = []
    fig = None

    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        for j in range(tries):
                action = policy.select_action(np.array(obs), expl_noise=0)
                obs, reward, done, info = env.step(action)
                avg_reward += reward
                avg_distance += info["distance"]
                pos_cyl = info["pos_cyl"]
                data_scatter["x"].append(pos_cyl[0])
                data_scatter["y"].append(pos_cyl[1])
                data_scatter["c"].append("red")
                if done == True:
                    data_scatter["c"][-1] = "green"
                    avg_success += 1
                    break
    
    
    avg_reward /= eval_episodes
    avg_success /= eval_episodes
    avg_distance /= eval_episodes
    data_scatter["title"] = "Threshold:{:.2f}m  Success rate:{:.3f}  Avg distance:{:.3f}m".format(env.threshold, avg_success, avg_distance)
    if plot_scatter==True:
        fig = scatter_plotter(data_scatter, show_plot=False)
    
    print("\nEvaluation over {:d} episodes. Avg reward:{:.3f} Success rate:{:.3f} Avg distance:{:.3f}".format(eval_episodes, avg_reward, avg_success, avg_distance))
    return avg_reward, avg_success, avg_distance, fig

def observe(env, replay_buffer, observation_steps, tries=1):
    """ Run episodes while taking random actions and filling replay_buffer
    
        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for
            tries (int): how many tries does the robot have in an episode
    
    """
    
    time_steps = 0
    obs = env.reset()
    done = False
    if tries==None:
        tries = observation_steps + 1 # no reset condition

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)

        replay_buffer.store(obs, action, new_obs, reward, done)

        obs = new_obs
        time_steps += 1

        if (done or ((time_steps % tries) == 0)):
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}. Cylinder position: {}. Reward: {:.3f}. Distance: {:.3f}".format(time_steps, 
            observation_steps, np.around(info["pos_cyl"],3), reward, info["distance"]), end="")
        sys.stdout.flush()
    print()
        

def train(agent, env, runner, replay_buffer, params, writer):
    """Train the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            runner (Runner): class to interact with the environment and step into new episode
            replay_buffer (ReplayBuffer): buffer to collect experience
            params (dict): Dictionary of training hyperparameters
            writer (SummaryWriter): tensorboard writer    
    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False 
    obs = env.reset()
    rewards = []
    episode_distances = []
    distances = []
    best_avg = -2000
    
    print("\n---------------------------------------")
    print("Training started for {} steps.".format(params["EXPLORATION"]))
    print("---------------------------------------")
    
    while total_timesteps < params["EXPLORATION"]:
    
         
        if total_timesteps != 0:             
            if (done or (episode_timesteps + 1) % params["TRIES"] == 0):
                rewards.append(episode_reward)
                distances.append(np.mean(episode_distances))
                avg_reward = np.mean(rewards[-100:])                

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)                
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)
                writer.add_scalar("avg_distance", np.mean(distances[-100:]), total_timesteps)
                writer.add_scalar("episode_distance", np.mean(episode_distances), total_timesteps)

                print("\rTimestep:{:d} Episode Num:{:d} Reward:{:f} Avg Reward:{:f} ".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    # print("saving best model....\n")
                    agent.save(params["FILENAME"],"saves")
                    params["EXPLORE_NOISE"] /= 1.01 # Exploration noise decay
                    
                if (avg_reward >= params["REWARD_THRESH"]) and (total_timesteps > 1000):
                    print("\nTarget average reward {} reached!".format(params["REWARD_THRESH"]))
                    break

                if not done:
                    runner.obs=env.reset()
                
                episode_reward = 0                
                episode_distances = []
                episode_timesteps = 0
                episode_num += 1             

            agent.train(replay_buffer, params["BATCH_SIZE"])

            # Evaluate episode
            if timesteps_since_eval >= params["EVAL_FREQUENCY"]:
                timesteps_since_eval %= params["EVAL_FREQUENCY"]
                eval_reward, eval_success, _, _ = evaluate_policy(agent, env, eval_episodes=params["EVAL_EPISODES"], tries=params["TRIES"])
                writer.add_scalar("eval_reward", eval_reward, total_timesteps)
                writer.add_scalar("eval_success", eval_success, total_timesteps)

                if best_avg < eval_reward:
                    best_avg = eval_reward
                    # print("saving best model....\n")
                    agent.save(params["FILENAME"],"saves")
                    params["EXPLORE_NOISE"] /= 1.01 # Exploration noise decay

                if eval_success >= 0.95: # threshold scheduling
                    agent.save(params["FILENAME"],"saves")
                    params["DISTANCE_THRESH"] -= 0.025
                    env.threshold = params["DISTANCE_THRESH"]
                    best_avg /= 2
                    print("Success rate reached 95%, distance threshold decreased to {:.3f}m.".format(env.threshold))


            
        reward, done, dist = runner.next_step(episode_timesteps, params["EXPLORE_NOISE"])
        episode_reward += reward
        episode_distances.append(dist)        

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        
    print("\n---------------------------------------")
    print("Training finished.")
    print("---------------------------------------")