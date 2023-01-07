from flappy import FlappyBird
import numpy as np
import matplotlib.pyplot as plt
from algorithms import SARSA


def evaluate_learning(sim, series_size=100, num_series=50, gamma=0.99, alpha=0.8, epsilon=0.001):

    # initialise Q values
    Qvalue = np.zeros((sim.num_states, sim.num_actions))
    #Qvalue = np.load('SARSA.npy')

    # graph
    fig, axes = plt.subplots()

    # total_reward_list and total_episodes
    total_reward_list = [0]
    total_episodes = 0

    # loop each series
    for series in range(num_series):
        print("series = %r/%d" % (series, num_series))  # Print the current stage
        reward_list = []
        # loop each episode in each series
        for episode in range(series_size):
            # training use Q learning policy
            Qvalue = SARSA(sim, gamma=gamma, alpha=alpha, epsilon=epsilon, Qtable=Qvalue)
            # add game score in reward list
            reward_list.append(sim.score)
            total_episodes += 1
        # initialise Q values
        newQvalue = Qvalue
        # get average score
        total_reward_list.append(np.mean(np.array(reward_list)))
        # save training network
        np.save('SARSA.npy', newQvalue)

    # number of episodes
    num_episodes = np.arange(0, total_episodes + 1, series_size)
    total_reward_list = np.array(total_reward_list)

    axes.plot(num_episodes, total_reward_list)
    axes.set_xlabel("Episodes")
    axes.set_ylabel("Average score")
    return


evaluate_learning(sim=FlappyBird())
plt.title('SARSA Average score per series')
plt.tight_layout()
plt.show()
