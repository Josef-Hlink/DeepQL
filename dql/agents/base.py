from collections import deque

from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class BaseAgent:
    """ Base functionality for all agents. """

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float,
        actionSpace: int, stateSpace: int
    ) -> None:

        self.getAction: function = {
            'e-greedy': self.epsilonGreedyAction,
            'boltzmann': self.boltzmannAction,
            'ucb': self.ucbAction
        }[explorationStrategy]

        self.eS = explorationStrategy
        self.eV = explorationValue
        self.alpha = alpha
        self.gamma = gamma
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        self.model = self.createModel()

    def createModel(self) -> Sequential:
        """ Creates and compiles a basic neural network. """
        model = Sequential()
        model.add(Dense(24, input_shape=(self.stateSpace,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.actionSpace, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    @property
    def epsilon(self) -> float: return self.eV if self.eS == 'e-greedy' else None
    
    @property
    def tau(self) -> float: return self.eV if self.eS == 'boltzmann' else None
    
    @property
    def zeta(self) -> float: return self.eV if self.eS == 'ucb' else None

    def anneal(self):

        self.eV = max(0.01, self.eV * 0.95)

    def epsilonGreedyAction(self, state) -> int:
        
        if np.random.rand() < self.epsilon: return np.random.randint(0, 2)
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
    
    def boltzmannAction(self, state) -> int:

        actionProbs = self.model.predict(self.model.predict(np.expand_dims(state, axis=0)), verbose=0)[0]
        actionProbs = actionProbs / self.tau
        actionProbs = actionProbs - np.max(actionProbs)
        actionProbs = np.exp(actionProbs)/np.sum(np.exp(actionProbs))

        return np.random.choice(np.arange(0, 2), p=actionProbs)

    def ucbAction(self, state) -> int:

        actionProbs = self.model.predict(self.model.predict(np.expand_dims(state, axis=0)), verbose=0)[0]
        actionProbs = actionProbs / self.tau
        actionProbs = actionProbs - np.max(actionProbs)
        actionProbs = np.exp(actionProbs)/np.sum(np.exp(actionProbs))
    
    def train(self, env: gym.Env, numEpisodes: int, episodeLength: int, V: bool, D: bool) -> list:

        global printV, printD
        printV, printD = PrintIfVerbose(V), PrintIfDebug(D)

        scores = []


        for i in prog(range(numEpisodes), V, f'Training {self.__class__.__name__}'):

            state, _ = env.reset()
            state = np.array(state)

            for j in range(episodeLength):

                # TODO: Implement model fitting
                ...

        return scores
