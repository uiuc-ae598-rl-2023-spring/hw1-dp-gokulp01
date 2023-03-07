from gridworld import GridWorld 
import random
from matplotlib import pyplot as plt
import numpy as np
from grid_plotter import *

env = GridWorld(hard_version=False)
env.reset()
algorithms = {
    0: "Policy Iteration",
    1: "Value Iteration",
    2: "SARSA",
    3: "Q-Learning"
}



class PolicyIteration:
    def __init__(self, env, theta=1e-6, gamma=0.95):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'V': [],
            'iters': []
        }
        
    def policy_iteration(self, plots=True):
        V = np.zeros(self.env.num_states)
        pi = np.ones([self.env.num_states, self.env.num_actions]) / self.env.num_actions
        iters = 0
        
        while True:
            delta = np.inf
            while delta > self.theta:
                delta = 0
                iters += 1
                for s in range(self.env.num_states):
                    v = 0
                    for a in range(self.env.num_actions):
                        q = 0
                        for s1 in range(self.env.num_states):
                            q += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * V[s1])
                        v += pi[s][a] * q
                    delta = max(delta, abs(v - V[s]))
                    V[s] = v
                    
            self.log['V'].append(np.mean(V))
            self.log['iters'].append(iters)
            
            policy_stable = True
            for s in range(self.env.num_states):
                chosen_a = np.argmax(pi[s])
                q_vals = [sum([self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * V[s1]) for s1 in range(self.env.num_states)]) for a in range(self.env.num_actions)]
                best_a = np.argmax(q_vals)
                if chosen_a != best_a:
                    policy_stable = False
                pi[s] = np.eye(self.env.num_actions)[best_a]
                
            if policy_stable:
                if plots:
                    plot_PI(env, pi, self.log)
                return V, np.argmax(pi, axis=1), self.log



class ValueIteration:
    def __init__(self, env, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.log = {
            't': [0],
            's': [],
            'a': [],
            'r': [],
            'V': [],
            'iters': []
        }
        
    def value_iteration(self, verbose=True, plots=True):
        V = [0] * self.env.num_states
        policy = [1] * self.env.num_states
        iters = 0
        while True:
            delta = 0
            for s in range(self.env.num_states):
                v = V[s]
                max_v = float('-inf')
                max_a = None
                for a in range(self.env.num_actions):
                    q = 0
                    for s1 in range(self.env.num_states):
                        q += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * V[s1])
                    if q > max_v:
                        max_v = q
                        max_a = a
                V[s] = max_v
                policy[s] = max_a
                delta = max(delta, abs(v - V[s]))
            iters += 1
            self.log['iters'].append(iters)
            self.log['V'].append(np.mean(V))
            if delta < self.theta:
                break
        
        print("Value Iteration")
        print("+++++")
        print("Generating plots")
        
        if plots:
            plot_VI(env, policy, self.log)
            print("Completed plot generation")
            print("+++++")
        return V, policy, self.log


def TD_0(pi, env, alpha=0.5, num_episodes = 5000):
    num_states = 25
    V = np.zeros(num_states)
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': []
    }
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V

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
            'iters': []
    }

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.env.num_actions)
        else:
            return self.q[state].index(max(self.q[state]))

    def learn(self, num_episodes=5000, plots=True):
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
        print("SARSA")
        print("+++++")
        print("Generating plots")
        if plots:
            plot_SARSA(self.env, pi, self.log, V_approx)
            print("Completed plot generation")
            print("+++++")
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
            'iters': []
    }
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.num_actions - 1)
        else:
            return np.argmax(self.Q[state, :])

    def learn(self, num_episodes=5000, plots=True):
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
        print("QLearning")
        print("+++++")
        print("Generating plots")
        if plots:
            plot_QL(self.env, pi, self.log, V_approx)
            print("Completed plot generation")
            print("+++++")
        return V_approx, self.Q, pi, self.log


def main():

    env = GridWorld(hard_version=False)
    env.reset()
    policy_iteration = PolicyIteration(env)
    V1, pi1, log1 = policy_iteration.policy_iteration()       

    value_iteration = ValueIteration(env)
    V2, pi2, log2 = value_iteration.value_iteration()   

    sarsa = SARSA(env)
    V3, q3, pi3, log3 = sarsa.learn(5000)

    q_learn = QLearning(env)
    V4, q4, pi4, log4 = q_learn.learn(5000)

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

