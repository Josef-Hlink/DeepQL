from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog
from dql.agents.base import BaseAgent

import numpy as np
from gym import Env


class TargetAgent(BaseAgent):

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingTemperature: float,
        actionSpace: int, stateSpace: int, V: bool, D: bool,
        updateFrequency: int
    ) -> None:
        
        super().__init__(
            explorationStrategy, alpha, gamma, annealingTemperature, explorationValue, actionSpace, stateSpace, V, D
        )

        global printV, printD
        printV, printD = PrintIfVerbose(V), PrintIfDebug(D)

        self.updateFrequency = updateFrequency
        self.targetModel = self.createModel()
    
    def updateTargetModel(self):
        self.targetModel.set_weights(self.model.get_weights())
    
    def learn(self, state, action, reward, nextState, done):
        target = reward
        if not done:
            target += self.gamma * np.amax(self.targetModel.predict(nextState.reshape(1, -1), verbose=0)[0])
        target_f = self.model.predict(state.reshape(1, -1), verbose=0)
        target_f[0][action] = target
        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def train(self, env: Env, numEpisodes: int, episodeLength: int, V: bool) -> list[int]:

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
                    self.anneal()
                    break

                self.learn(state, action, reward, nextState, done)
                state = nextState

            if i % self.updateFrequency == 0: self.updateTargetModel()
        
        env.close()

        return scores
