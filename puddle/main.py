
import gymnasium as gym
import gym_puddle, warnings 
import matplotlib.pyplot as plt
import numpy as np
import json

warnings.filterwarnings("ignore")


# Hyperparameters
JSON_FILE = "./env_configs/pw1.json"
NUM_EPISODES = 20000
EPSILON = 1
EPSILON_DECAY_RATE = 0.9995
MIN_EPSILON = 0.01
LEARNING_RATE = 0.02
GAMMA = 0.99
class QLearningAgent:
    def init(self, env, state_grid, alpha=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE, min_epsilon=MIN_EPSILON):
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_size = self.env.action_space.n  
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = self.initial_epsilon = epsilon  
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)


    def preprocess_state(self, state):
        return tuple(discretize(state, self.state_grid))


    def reset_episode(self, state):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action


    def reset_exploration(self, epsilon=None):
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon


    def act(self, state, reward=None, done=None, mode='train'):
        state = self.preprocess_state(state)
        if mode == 'test':
            action = np.argmax(self.q_table[state])
        else:
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])
            action = np.random.randint(0, self.action_size) if np.random.uniform(0, 1) < self.epsilon else np.argmax(self.q_table[state])
        self.last_state = state
        self.last_action = action
        return action


def create_uniform_grid(low, high, bins=(20, 20)):
    return [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]


def discretize(sample, grid):
    return [int(np.digitize(s, g)) for s, g in zip(sample, grid)]


def run(agent, env, num_episodes=NUM_EPISODES, mode='train'):
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        state = env.reset()[0]
        action = agent.reset_episode(state)
        total_reward = 0
        done = False
        while not done:
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)
        scores.append(total_reward)
        if mode == 'train' and i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            max_avg_score = max(max_avg_score, avg_score)
            print(f"Episode {i_episode}/{num_episodes} | Max Avg Score: {max_avg_score}", end="\r")
    env.close()
    return scores


def main():

    with open(JSON_FILE) as f:
        setupEnv = json.load(f)

    env = gym.make(
    "PuddleWorld-v0",
    start=setupEnv["start"],
    goal=setupEnv["goal"],
    goal_threshold=setupEnv["goal_threshold"],
    noise=setupEnv["noise"],
    thrust=setupEnv["thrust"],
    puddle_top_left=setupEnv["puddle_top_left"],
    puddle_width=setupEnv["puddle_width"],
    )
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high)
    q_agent = QLearningAgent(env, state_grid)
    scores = run(q_agent, env)
    plt.plot(scores)
    plt.title("Scores")
    plt.show()
main()