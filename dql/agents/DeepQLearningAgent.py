from collections import deque
import numpy as np

import gym


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam

class DeepQLearningAgent:

    def __init__(self,StateSize, ActionSize):
        self.learningRate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batchSize = 32

        self.actionSize = ActionSize
        self.stateSize = StateSize

        self.memory = deque(maxlen=2000)


        #neural network with 4 inputs, 2 hidden layers with 24 nodes each, and 2 outputs
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(1,4), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))


    def epsilonGreedyAction(self, state):

        if np.random.rand() < self.epsilon:

            return np.random.randint(0, self.actionSize)
        

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, [1, 1, 4])


        return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def epsilonAnneal(self):

        if self.epsilon > 0.01:
            self.epsilon *= 0.90
    
    def remember(self, state, action, reward, nextState, done):

        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        
        self.memory.append({ 'currentState': state, 'action': action, 'reward': reward, 'nextState': nextState, 'done': done })

    def replay(self):
        
        if len(self.memory) < self.batchSize:
            return

        batch = np.random.choice(self.memory, self.batchSize)

        for item in batch:

            state = item['currentState']
            action = item['action']
            reward = item['reward']
            nextState = item['nextState']
            done = item['done']

            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.reshape(state, [1, 1, 4])

            nextState = tf.convert_to_tensor(state, dtype=tf.float32)
            nextSstate = tf.reshape(nextState, [1, 1, 4])

            target_f = self.model.predict(state, verbose=0)
            target = reward

            if not done:

                target = reward + self.gamma * np.amax(self.model.predict(nextState, verbose=0)[0])

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

def main():

    env = gym.make('CartPole-v1', render_mode='human')

    StateSize = env.observation_space.shape[0]
    # print("State size: " + str(StateSize))
    ActionSize = env.action_space.n

    episodes = 20
    maxIterations = 500 #max of cartpole

    Agent = DeepQLearningAgent(StateSize, ActionSize)

    for i in range(episodes):

        print("Episode: {}/{}".format(i, episodes))


        state, _ = env.reset()
        state = np.array(state)

        for j in range(maxIterations):

            env.render()

            action = Agent.epsilonGreedyAction(state)

            nextState, reward, done, _, __ = env.step(action)

            nextState = np.array(nextState).astype(np.float32)

            Agent.remember(state, action, reward, nextState, done)

            state = nextState

            if done:
                print("Episode: {}/{}, score: {}, e: {:.2}".format(i, episodes, j, Agent.epsilon))
                Agent.epsilonAnneal()
                break

        Agent.replay()

if __name__ == "__main__":
    main()