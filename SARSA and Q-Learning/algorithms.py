import numpy as np


# epsilon greedy policy
def epsilon_greedy(epsilon, Qtable, state):
    num_actions = Qtable.shape[1]
    if np.random.random() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Qtable[state, :])
    return action


def Q_learning(env, gamma, alpha, epsilon, Qtable, max_steps=np.inf):

    # initialise state
    current_position = env.reset()
    steps = 0

    while not env.is_terminal() and steps < max_steps:
        # Choose A form S using greedy policy
        action = epsilon_greedy(epsilon, Qtable, current_position)

        # get the next position and rewards
        next_position, reward = env.next(action)

        # update Q-learning rule
        Qtable[current_position, action] += alpha * (
                reward + gamma * np.max(Qtable[next_position, :]) - Qtable[current_position, action])

        current_position = next_position
        steps += 1

    # return the policy
    return Qtable


def SARSA(env, gamma, alpha, epsilon, Qtable, max_steps=np.inf):

    # initialise state
    current_position = env.reset()
    steps = 0

    # Choose A form S using greedy policy
    action = epsilon_greedy(epsilon, Qtable, current_position)

    while not env.is_terminal() and steps < max_steps:
        # get the next position and rewards
        next_position, reward = env.next(action)

        # choose A' from S' using soft greedy policy
        next_action = epsilon_greedy(epsilon, Qtable, next_position)

        # update SARSA rule
        Qtable[current_position, action] += alpha * (
                reward + gamma * Qtable[next_position, next_action] - Qtable[current_position, action])

        # update S' to S and A' to A
        current_position = next_position
        action = next_action
        steps += 1

    # return the policy
    return Qtable
