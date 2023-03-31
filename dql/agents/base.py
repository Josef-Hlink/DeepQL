from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class BaseAgent:
    """ Base functionality for all agents. """

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingTemperature: float,
        actionSpace: int, stateSpace: int, V: bool, D: bool
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
        self.annealTemp = annealingTemperature
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        
        global printV, printD
        printV, printD = PrintIfVerbose(V), PrintIfDebug(D)
        
        self.model = self.createModel()
        self.counts = np.zeros(self.actionSpace)

    def createModel(self) -> Sequential:
        """ Creates and compiles a basic neural network. """
        model = Sequential()
        model.add(Dense(64, input_shape=(self.stateSpace,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actionSpace, activation='linear'))

        # set random weights initialization to He initialization
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights: continue
            weights[0] = np.random.randn(*weights[0].shape) * np.sqrt(2 / weights[0].shape[0])
            layer.set_weights(weights)

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    @property
    def epsilon(self) -> float: return self.eV if self.eS == 'e-greedy' else None
    
    @property
    def tau(self) -> float: return self.eV if self.eS == 'boltzmann' else None
    
    @property
    def zeta(self) -> float: return self.eV if self.eS == 'ucb' else None

    def anneal(self):

        if self.zeta is None: self.eV = max(0.01, self.eV * self.annealTemp)

    def epsilonGreedyAction(self, state) -> int:

        if np.random.rand() < self.epsilon: return np.random.randint(0, 2)
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
    
    def boltzmannAction(self, state) -> int:

        Q = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        Q /= self.tau; Q -= np.max(Q)
        return np.random.choice(self.actionSpace, p=np.exp(Q) / np.sum(np.exp(Q)))

    def ucbAction(self, state) -> int:

        Q = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        Q += np.sqrt(self.zeta * np.log(np.sum(self.counts)) / (self.counts+0.001))
        printD(f'Q: {Q}, t: {np.sum(self.counts)}, counts: {self.counts}')
        return np.argmax(Q)

    def learn(self, state, action, reward, nextState, done) -> None:
        
        target = reward
        if not done: target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(nextState, axis=0), verbose=0)[0])
        targetF = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        targetF[0][action] = target
        state = np.expand_dims(state, axis=0)
        self.model.fit(state, targetF, epochs=1, verbose=0)
    
    def train(self, env: gym.Env, numEpisodes: int, episodeLength: int, V: bool) -> list[int]:

        scores = []

        for i in prog(range(numEpisodes), V, f'Training {self.__class__.__name__}'):

            state, _ = env.reset()
            state = np.array(state)

            for j in range(episodeLength):
                
                action = self.getAction(state)
                self.counts[action] += 1
                nextState, reward, done, timedOut, _ = env.step(action)

                if done or timedOut:
                    scores.append(j)
                    break

                self.learn(state, action, reward, nextState, done)
                state = nextState

            self.anneal()

        env.close()
        return scores
