from collections import deque

from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog

import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam

class DeepQLearningAgent:

    def __init__(self,
        explorationStrategy: str,
        alpha: float, gamma: float, explorationValue: float,
        batchSize: int, memorySize: int,
        actionSpace: int, stateSpace: int
    ) -> None:

        self.takeAction: function = {
            'e-greedy': self.epsilonGreedyAction,
            'boltzmann': self.boltzmannAction,
            'ucb': self.ucbAction
        }[explorationStrategy]

        self.explorationStrategy = explorationStrategy
        self.alpha = alpha
        self.gamma = gamma
        self.explorationValue = explorationValue
        self.batchSize = batchSize
        self.maxMemorySize = memorySize
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace

        self.memory = deque(maxlen=memorySize)
        self._initializeModel()
        self._setExplorationValues()


    def _initializeModel(self) -> None:

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.stateSpace,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.actionSpace, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))

    def _setExplorationValues(self) -> None:
        self.epsilon = self.explorationValue if self.explorationStrategy == 'e-greedy' else None
        self.tau = self.explorationValue if self.explorationStrategy == 'boltzmann' else None
        self.zeta = self.explorationValue if self.explorationStrategy == 'ucb' else None
        return

    def epsilonGreedyAction(self, state):

        if np.random.rand() < self.epsilon:
            
            return np.random.randint(0, 2)
        
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, [1, 4])

        return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def boltzmannAction(self, state):

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, [1, 4])

        actionProbs = self.model.predict(state, verbose=0)[0]
        actionProbs = actionProbs / self.tau
        actionProbs = actionProbs - np.max(actionProbs)
        actionProbs = np.exp(actionProbs)/np.sum(np.exp(actionProbs))

        return np.random.choice(np.arange(0, 2), p=actionProbs)
    
    def ucbAction(self, state):

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, [1, 4])

        actionValues = self.model.predict(state, verbose=0)[0]
        actionValues = actionValues + np.sqrt(np.log(self.zeta) / self.zeta + 1)

        return np.argmax(actionValues)

    def anneal(self):


        if self.explorationStrategy == 'e-greedy':
            if self.epsilon > 0.01:
                self.epsilon *= 0.995
        elif self.explorationStrategy == 'boltzmann':
            if self.tau > 0.01:
                self.tau *= 0.995
        elif self.explorationStrategy == 'ucb':
            if self.zeta > 0.01:
                self.zeta *= 0.995
    
    def remember(self, state, action, reward, nextState, done):

        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        
        self.memory.append({'currentState': state, 'action': action, 'reward': reward, 'nextState': nextState, 'done': done})

    def replay(self):
        
        if len(self.memory) < self.batchSize:
            return

        batch = np.random.choice(self.memory, self.batchSize)

        states = np.array([item['currentState'] for item in batch])
        actions = np.array([item['action'] for item in batch])
        rewards = np.array([item['reward'] for item in batch])
        nextStates = np.array([item['nextState'] for item in batch])
        dones = np.array([item['done'] for item in batch])

        target_f = self.model.predict(states, verbose=0)
        target = rewards
        
        notDones = np.logical_not(dones)
        target[notDones] += self.gamma * np.amax(self.model.predict(nextStates[notDones], verbose=0), axis=1)
        
        target_f[np.arange(self.batchSize), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)


def run(args: dict[str, any]) -> None:

    V, D = args['verbose'], args['debug']

    global printV, printD
    printV, printD = PrintIfVerbose(V), PrintIfDebug(D)

    explorationStrategy = args['explorationStrategy']
    alpha, gamma, explorationValue = args['alpha'], args['gamma'], args['explorationValue']
    batchSize, memorySize = args['batchSize'], args['memorySize']

    renderMode = 'human' if False else 'none'

    env = gym.make('CartPole-v1', render_mode=renderMode)

    actionSpace = env.action_space.n
    observationSpace = env.observation_space.shape[0]

    scores = []

    agent = DeepQLearningAgent(
        explorationStrategy, alpha, gamma, explorationValue, batchSize, memorySize, actionSpace, observationSpace
    )

    numEpisodes = args['numEpisodes']
    episodeLength = args['episodeLength']

    env.reset()

    for i in prog(range(numEpisodes), V, 'Training'):

        state, _ = env.reset()
        state = np.array(state)

        for j in range(episodeLength):


            ## TODO: Add render flag
            # if V:
            #     env.render()

            if explorationStrategy == 'e-greedy':
                action = agent.epsilonGreedyAction(state)
            elif explorationStrategy == 'boltzmann':
                action = agent.boltzmannAction(state)
            elif explorationStrategy == 'ucb':
                action = agent.ucbAction(state)

            nextState, reward, done, timedOut, _ = env.step(action)

            nextState = np.array(nextState).astype(np.float32)

            agent.remember(state, action, reward, nextState, done)

            if done or timedOut:

                scores.append(j)
                agent.anneal()
                #env.reset()
                break

            state = nextState

        agent.replay()

    printV(scores)


if __name__ == "__main__":
    run()
