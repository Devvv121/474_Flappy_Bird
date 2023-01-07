#!/usr/bin/env python
from collections import deque
import sys
import tensorflow as tf
import random
import cv2
import wrapped_flappy_bird as game # Environment
import numpy as np

sys.path.append("/")
tf.compat.v1.disable_eager_execution()

#All the arguments use for trainning
Alpha=1e-6
explore = 1000000 # frames to anneal epsilon
gamma = 0.9 # decay rate
actions = 2  # actions [Stay,up]
observe = 100000  # timesteps to observe before training
e = 0.0001  # starting value of epsilon
final_e=0.00001
RM = 50000 # number of previous transitions to remember
batch = 100 # size of Sample
Game= 'bird' # the name of the game for log files

def training(s, figerout, networks):
    # store the previous observations in replay memory
    P = deque()

    # open game
    States = game.GameState()

    # cost function
    a = tf.compat.v1.placeholder("float", [None, actions])
    R_A = tf.reduce_sum(tf.multiply(figerout, a), 1)
    b = tf.compat.v1.placeholder("float", [None])
    loss = tf.reduce_mean(tf.square(b - R_A))

    step_train = tf.compat.v1.train.AdamOptimizer(Alpha).minimize(loss)

    # get the first state and preprocess the image to 80x80x4
    save = np.zeros(actions)

    save[0] = 1
    p_t, w_0, T = States.frame_step(save)
    p_t = cv2.cvtColor(cv2.resize(p_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    S = np.stack((p_t, p_t, p_t, p_t), 2)
    ret, p_t = cv2.threshold(p_t,1,255,cv2.THRESH_BINARY)


    # load network
    files = tf.train.get_checkpoint_state("SAN")
    saver = tf.compat.v1.train.Saver()
    networks.run(tf.compat.v1.global_variables_initializer())
    if files and files.model_checkpoint_path:
        saver.restore(networks, files.model_checkpoint_path)
        print("find a DQN:", files.model_checkpoint_path)
    else:
        print("There is no any old network")

    epsilon = e
    timestep = 0
    while "A" != "B":
        #training
        a_step = np.zeros([actions])
        read_t = figerout.eval({s : [S]})[0]
        # e-greedy
        if random.random() <= epsilon:
            action_i = random.randrange(actions)
            a_step[action_i] = 1
        else:
            #greedy
            action_i = np.argmax(read_t)
            a_step[action_i] = 1

        #decrease e
        if epsilon > final_e and timestep > observe:
            epsilon -= (e - final_e) / explore

        # selected action and observe R1 and S1
        p_t1_col, r_t, T = States.frame_step(a_step)

        p_t1 = cv2.cvtColor(cv2.resize(p_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)

        ret, p_t1 = cv2.threshold(p_t1, 1, 255, cv2.THRESH_BINARY)

        p_t1 = np.reshape(p_t1, (80, 80, 1))

        S1 = np.append(p_t1, S[:, :, :3], 2)

        # store transition
        P.append((S, a_step, r_t, S1, T))

        if len(P) > RM:
            P.popleft()

        if timestep > observe:
            # sample a smallbatch of size 64
            Sample = random.sample(P, batch)
            # get all variables
            y_batch = []
            s_P_batch = [d[0] for d in Sample]
            s_P1_batch = [d[3] for d in Sample]
            readout_P1_batch = figerout.eval({s : s_P1_batch})
            a_batch = [d[1] for d in Sample]
            r_batch = [d[2] for d in Sample]


            for i in range(0, len(Sample)):
                T = Sample[i][4]
                # improve policy
                if not T:
                    y_batch.append(r_batch[i] + gamma * np.max(readout_P1_batch[i]))
                else :
                    y_batch.append(r_batch[i])

            # training the network
            step_train.run( {b : y_batch,a : a_batch,s : s_P_batch})

        # update
        S = S1
        timestep += 1

        # save progress every 10000 iterations
        if timestep % 10000 == 0:
            print("save files")
            saver.save(networks, 'SAN/' + Game + '-dqn', timestep)

        print("step", timestep, "/ State","/ epsilon", epsilon,"/ action", action_i, "/ R", r_t,"/ Q_max %e" % np.max(read_t))

def maxpooling_2x2(A):
    return tf.nn.max_pool(A,[1, 2, 2, 1],  [1, 2, 2, 1], "SAME")

def weight(S):
    initial = tf.random.truncated_normal(S, 0.01)
    return tf.Variable(initial)

def bias(S):
    initial = tf.constant(0.01, shape = S)
    return tf.Variable(initial)

def c2d(S, W, stride):
    return tf.nn.conv2d(S, W, [1, stride, stride, 1], "SAME")

def buildnetwork():
    W1 = weight([8, 8, 4, 32])
    b1 = bias([32])

    W2 = weight([4, 4, 32, 64])
    b2 = bias([64])

    W3 = weight([3, 3, 64, 64])
    b3 = bias([64])

    wfc1 = weight([1600, 512])
    bfc1 = bias([512])

    wfc2 = weight([512, actions])
    bfc2 = bias([actions])

    # input-layer
    input = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    # hidden-layers
    conv1 = tf.nn.relu(c2d(input, W1, 4) + b1)
    pool1 = maxpooling_2x2(conv1)

    conv2 = tf.nn.relu(c2d(pool1, W2, 2) + b2)

    conv3 = tf.nn.relu(c2d(conv2, W3, 1) + b3)

    conv3_f = tf.reshape(conv3, [-1, 1600])

    h_fc = tf.nn.relu(tf.matmul(conv3_f, wfc1) + bfc1)

    # readout——layer
    output = tf.matmul(h_fc, wfc2) + bfc2

    return input, output



def training_bird():
    networks=tf.compat.v1.InteractiveSession()
    input, output = buildnetwork()
    training(input, output, networks)

training_bird()


