from matplotlib import pyplot as plt
import numpy as np
from train_gridworld import SARSA, QLearning
algorithms = {
    0: "Policy Iteration",
    1: "Value Iteration",
    2: "SARSA",
    3: "Q-Learning"
}

def plot_VI(env, policy, log):
    sp = env.reset()
    log['s'].append(sp)
    done = False
    while not done:
        a = policy[sp]
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(log['iters'],log['V'])
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Mean(V)")
        ax1.set_title("Learning curve: VI")
        ax1.grid(True)

        ax2.plot(log['t'], log['s'])
        ax2.plot(log['t'][:-1], log['a'])
        ax2.plot(log['t'][:-1], log['r'])
        ax2.set_title("State, Action, Reward: VI")
        ax2.set_xlabel("Time")
        ax2.legend(['s', 'a', 'r'])
        ax2.grid(True)

        fig.subplots_adjust(hspace=0.4)

        plt.savefig('figures/gridworld/learning_and_trajectory_vi.png')


def plot_SARSA(env, pi, log, V_approx):
    sp = env.reset()
    log['s'].append(sp)
    done = False
    while not done:
        a = np.argmax(pi[sp])
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    ax1.plot(log['t'], log['s'])
    ax1.plot(log['t'][:-1], log['a'])
    ax1.plot(log['t'][:-1], log['r'])
    ax1.set_title("State, Action, Reward: SARSA")
    ax1.set_xlabel("Time")
    ax1.legend(['s', 'a', 'r'])
    ax1.grid(True)

    ax2.plot(log['episodes'],log['G'])
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Total Return (G)")
    ax2.set_title("Learning curve: SARSA")
    ax2.grid(True)

    ax3.plot(V_approx)
    ax3.set_xlabel("States")
    ax3.set_ylabel("Value Function")
    ax3.set_title("State-Value Function Learned by TD(0): SARSA")
    ax3.grid(True)


    fig.subplots_adjust(hspace=0.4)

    plt.savefig('figures/gridworld/sarsa_plots.png')
      

def plot_QL(env, pi, log, V_approx):
    sp = env.reset()
    log['s'].append(sp)
    done = False
    while not done:
        a = np.argmax(pi[sp])
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)
    
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

        ax1.plot(log['t'], log['s'])
        ax1.plot(log['t'][:-1], log['a'])
        ax1.plot(log['t'][:-1], log['r'])
        ax1.set_title("State, Action, Reward: Q Learning")
        ax1.set_xlabel("Time")
        ax1.legend(['s', 'a', 'r'])
        ax1.grid(True)

        ax2.plot(log['episodes'],log['G'])
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Total Return (G)")
        ax2.set_title("Learning curve: Q Learning")
        ax2.grid(True)

        ax3.plot(V_approx)
        ax3.set_xlabel("States")
        ax3.set_ylabel("Value Function")
        ax3.set_title("State-Value Function Learned by TD(0): Q Learning")
        ax3.grid(True)

        fig.subplots_adjust(hspace=0.4)

        plt.savefig('figures/gridworld/qlearning_plots.png')


def plot_alpha_sweep(algo_name, env, alpha_vals, filename):
    plt.figure()
    for alpha in alpha_vals:
        if algo_name == "SARSA":
            algo = SARSA(env, alpha=alpha)
        elif algo_name == "Q-Learning":
            algo = QLearning(env, alpha=alpha)
        V, Q, pi, log = algo.learn(plots=False)
        plt.plot(log['episodes'], log['G'], label=f"Alpha={alpha}")
    plt.title(f"Learning curve for different alpha: {algo_name}")
    plt.legend()
    plt.savefig(f"figures/gridworld/{filename}")

def plot_epsilon_sweep(algo_name, env, epsilon_vals, filename):
    plt.figure()
    for epsilon in epsilon_vals:
        if algo_name == "SARSA":
            algo = SARSA(env, epsilon=epsilon)
        elif algo_name == "Q-Learning":
            algo = QLearning(env, epsilon=epsilon)
        V, Q, pi, log = algo.learn(plots=False)
        plt.plot(log['episodes'], log['G'], label=f"Epsilon={epsilon}")
    plt.title(f"Learning curve for different epsilon: {algo_name}")
    plt.legend()
    plt.savefig(f"figures/gridworld/{filename}")

