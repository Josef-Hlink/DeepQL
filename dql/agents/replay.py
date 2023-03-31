from collections import deque

from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog
from dql.agents.base import BaseAgent

import numpy as np
from gym import Env


class ReplayAgent(BaseAgent):
    
    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingTemperature: float,
        actionSpace: int, stateSpace: int,
        batchSize: int, memorySize: int
    ) -> None:
        
        super().__init__(explorationStrategy, alpha, gamma, annealingTemperature, explorationValue, actionSpace, stateSpace)

        self.batchSize = batchSize
        self.maxMemorySize = memorySize
        self.memory = deque(maxlen=memorySize)

    def remember(self, state, action, reward, nextState, done):

        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        
        self.memory.append({'currentState': state, 'action': action, 'reward': reward, 'nextState': nextState, 'done': done})

    def learn(self):
        
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


    def train(self, env: Env, numEpisodes: int, episodeLength: int, V: bool, D: bool) -> list[int]:

        global printV, printD
        printV, printD = PrintIfVerbose(V), PrintIfDebug(D)

        scores = []

        for i in prog(range(numEpisodes), V, f'Training {self.__class__.__name__}'):

            state, _ = env.reset()
            state = np.array(state)

            for j in range(episodeLength):

                action = self.getAction(state)
                nextState, reward, done, timedOut, _ = env.step(action)
                self.remember(state, action, reward, nextState, done)

                if done or timedOut:
                    scores.append(j)
                    self.anneal()
                    break

                state = nextState

            self.learn()

        env.close()

        return scores
