from matplotlib import pyplot as plt
import numpy as np
from train_pendulum import SARSA, QLearning, cal_theta

algorithms = {
    0: "Policy Iteration",
    1: "Value Iteration",
    2: "SARSA",
    3: "Q-Learning"
}

def plot_SARSA(env, pi, log, V_approx):
    sp = env.reset()
    log['s'].append(sp)
    log['theta'].append(cal_theta(env.x[0]))
    log['thetadot'].append(env.x[1])
    done = False
    while not done:
        a = np.argmax(pi[sp])
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)
        log['theta'].append(cal_theta(env.x[0]))
        log['thetadot'].append(env.x[1])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(log['t'], log['s'])
    axs[0, 0].plot(log['t'][:-1], log['a'])
    axs[0, 0].plot(log['t'][:-1], log['r'])
    axs[0, 0].set_title("State, Action, Reward: SARSA")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].legend(['s', 'a', 'r'])

    axs[0, 1].plot(log['t'], log['theta'])
    axs[0, 1].plot(log['t'], log['thetadot'])
    axs[0, 1].axhline(y=np.pi, color='r', linestyle='-')
    axs[0, 1].axhline(y=-np.pi, color='r', linestyle='-')
    axs[0, 1].set_title("Theta, Theta_dot vs Time: SARSA")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].legend(['theta', 'theta_dot', 'Theta = pi', 'Theta = -pi'])

    axs[1, 0].plot(log['episodes'], log['G'])
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Total Return (G)")
    axs[1, 0].set_title("Learning curve: SARSA")

    axs[1, 1].plot(V_approx)
    axs[1, 1].set_xlabel("States")
    axs[1, 1].set_ylabel("Value Function")
    axs[1, 1].set_title("State-Value Function Learned by TD(0): SARSA")

    fig.suptitle('SARSA')

    plt.tight_layout()

    plt.savefig('figures/pendulum/sarsa_plots.png')


def plot_QL(env, pi, log, V_approx):

    sp = env.reset()
    log['s'].append(sp)
    log['theta'].append(cal_theta(env.x[0]))
    log['thetadot'].append(env.x[1])
    done = False
    while not done:
        a = np.argmax(pi[sp])
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)
        log['theta'].append(cal_theta(env.x[0]))
        log['thetadot'].append(env.x[1])
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(log['t'], log['s'])
    axs[0, 0].plot(log['t'][:-1], log['a'])
    axs[0, 0].plot(log['t'][:-1], log['r'])
    axs[0, 0].set_title("State, Action, Reward: Q-Learning")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].legend(['s', 'a', 'r'])

    axs[0, 1].plot(log['t'], log['theta'])
    axs[0, 1].plot(log['t'], log['thetadot'])
    axs[0, 1].axhline(y=np.pi, color='r', linestyle='-')
    axs[0, 1].axhline(y=-np.pi, color='r', linestyle='-')
    axs[0, 1].set_title("Theta, Theta_dot vs Time: Q-Learning")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].legend(['theta', 'theta_dot', 'Theta = pi', 'Theta = -pi'])

    axs[1, 0].plot(log['episodes'],log['G'])
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Total Return (G)")
    axs[1, 0].set_title("Learning curve: Q-Learning")

    axs[1, 1].plot(V_approx)
    axs[1, 1].set_xlabel("States")
    axs[1, 1].set_ylabel("Value Function")
    axs[1, 1].set_title("State-Value Function Learned by TD(0): Q-Learning")

    plt.tight_layout()

    plt.savefig('figures/pendulum/subplots_qlearning.png')



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
    plt.savefig(f"figures/pendulum/{filename}")

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
    plt.savefig(f"figures/pendulum/{filename}")

