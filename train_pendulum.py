from discrete_pendulum import Pendulum
import random
from matplotlib import pyplot as plt
import numpy as np
from pendulum_plotter import *


env = Pendulum() 
env.reset()
algorithms = {
    0: "Policy Iteration",
    1: "Value Iteration",
    2: "SARSA",
    3: "Q-Learning"
}

def TD_0(pi, env, alpha=0.5, num_episodes = 700):
    V = np.zeros(env.num_states)
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': [],
        'theta': [],
        'thetadot': []
    }
    for episode in range(num_episodes):
        s = env.reset()
        log['s'].append(s)
        log['theta'].append(cal_theta(env.x[0]))
        log['thetadot'].append(env.x[1])
        done = False
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(cal_theta(env.x[0]))
            log['thetadot'].append(env.x[1])
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V
def cal_theta(x):
    theta = ((x + np.pi) % (2 * np.pi)) - np.pi
    return theta
class SARSA:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = [[0.0 for _ in range(self.env.num_actions)] for _ in range(self.env.num_states)]
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'G': [],
            'episodes': [],
            'iters': [],
            'theta': [],
            'thetadot': []
    }
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.env.num_actions)
        else:
            return self.q[state].index(max(self.q[state]))

    def learn(self, num_episodes=700, plots=True):
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions

        returns = []
        for i in range(num_episodes):
            s = self.env.reset()
            a = self.act(s)
            iters=0
            episode_return = 0.0
            for t in range(self.env.max_num_steps):
                iters+=1
                s_next, reward, done = self.env.step(a)
                a_next = self.act(s_next)
                episode_return += reward*self.gamma**(iters-1)
                self.q[s][a] += self.alpha * (reward + self.gamma * self.q[s_next][a_next] - self.q[s][a])
                s = s_next
                a = a_next
                if done:
                    pi[s] = np.eye(self.env.num_actions)[np.argmax(self.q[s])]
                    self.log['G'].append(episode_return)
                    self.log['episodes'].append(i)
                    # returns.append(episode_return)
                    break
        V_approx = TD_0(pi, env)
        if plots == True:
            plot_SARSA(self.env, pi, self.log, V_approx)
        return V_approx, self.q, pi, self.log

    
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))

        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'G': [],
            'episodes': [],
            'iters': [],
            'theta': [],
            'thetadot': []
    }
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.num_actions - 1)
        else:
            return np.argmax(self.Q[state, :])

    def learn(self, num_episodes=700, plots=True):
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions

        episode_returns = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            iters=0
            while not done:
                iters+=1
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode_return += reward*self.gamma**(iters-1)
                next_action = np.argmax(self.Q[next_state, :])
                td_target = reward + self.gamma * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                state = next_state
            pi[state] = np.eye(self.env.num_actions)[np.argmax(self.Q[state])]
            # episode_returns.append(episode_return)
            self.log['G'].append(episode_return)
            self.log['episodes'].append(episode)
        V_approx = TD_0(pi, env)
        if plots == True:
            plot_QL(self.env, pi, self.log, V_approx)
        return V_approx, self.Q, pi, self.log



def main():
    env = Pendulum()
    env.reset()

    sarsa = SARSA(env)
    V3, q3, pi3, log3 = sarsa.learn(700)

    q_learn = QLearning(env)
    V4, q4, pi4, log4 = q_learn.learn(700)
    alpha_vals = np.linspace(0, 1, 11)
    epsilon_vals = np.linspace(0, 0.5, 11)

    for i in range(2):
        algo_name = algorithms[i + 2]
        if i == 0:
            plot_alpha_sweep(algo_name, env, alpha_vals, "alpha_sweep_SARSA.png")
            plot_epsilon_sweep(algo_name, env, epsilon_vals, "epsilon_sweep_SARSA.png")
        else:
            plot_alpha_sweep(algo_name, env, alpha_vals, "alpha_sweep_qlearning.png")
            plot_epsilon_sweep(algo_name, env, epsilon_vals, "epsilon_sweep_qlearning.png")

if __name__ == '__main__':
    main()


