from discrete_pendulum import Pendulum
import random
from matplotlib import pyplot as plt
import numpy as np


def TD_0(pi, alpha=0.5, num_episodes = 1000):
    V = np.zeros(env.num_states)
    
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
            'iters': [],
            'theta': [],
            'thetadot': []
        }

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.env.num_actions)
        else:
            return self.q[state].index(max(self.q[state]))

    def learn(self, num_episodes):
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions

        returns = []
        for i in range(num_episodes):
            s = self.env.reset()
            a = self.act(s)
            episode_return = 0.0
            for t in range(self.env.max_num_steps):
                s_next, reward, done = self.env.step(a)
                a_next = self.act(s_next)
                episode_return += reward
                self.q[s][a] += self.alpha * (reward + self.gamma * self.q[s_next][a_next] - self.q[s][a])
                s = s_next
                a = a_next
                if done:
                    pi[s] = np.eye(self.env.num_actions)[np.argmax(self.q[s])]

                    returns.append(episode_return)
                    break
        V_approx = TD_0(pi)
        print(V_approx)
        return returns, V_approx, self.q, pi

    
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.num_actions - 1)
        else:
            return np.argmax(self.Q[state, :])

    def learn(self, num_episodes):
        pi = np.ones((self.env.num_states,self.env.num_actions))/self.env.num_actions

        episode_returns = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode_return += reward
                next_action = np.argmax(self.Q[next_state, :])
                td_target = reward + self.gamma * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                state = next_state
            pi[state] = np.eye(self.env.num_actions)[np.argmax(self.Q[state])]
            episode_returns.append(episode_return)
        V_approx = TD_0(pi)
        print(V_approx)
        return episode_returns, V_approx, self.Q, pi




env = Pendulum(n_theta=15, n_thetadot=21)
env.reset()



print("---------------")
print("SARSA")
sarsa = SARSA(env)
returns, V_approx, q, pi = sarsa.learn(100)
print(f"Q = {q}")
print(f"V = {np.max(q,axis=1)}")
print(f"Policy = {np.argmax(pi,axis=1)}")
print(f"Approximate V using TD(0) = {V_approx}")



# plt.plot(returns)
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.show()


print("---------------")
print("QLearning")
q_learn = QLearning(env)
returns, V_approx, q, pi = q_learn.learn(100)
print(f"Q = {q}")
print(f"V = {np.max(q,axis=1)}")
print(f"Policy = {np.argmax(pi,axis=1)}")
print(f"Approximate V using TD(0) = {V_approx}")


# # plot the returns
# plt.plot(episode_returns)
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.show()


