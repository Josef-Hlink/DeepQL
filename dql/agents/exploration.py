""" Different exploration strategies that determine the action selection process of an agent. """

from abc import abstractmethod
from typing import Optional

from dql.agents.annealing import AnnealingScheme

import numpy as np


class ExplorationStrategy:
    """ Abstract base class for all exploration strategies. """

    @abstractmethod
    def __init__(self, annealingScheme: AnnealingScheme) -> None:
        """ Initializes an exploration strategy instance based on an annealing scheme. """
        self.sV = annealingScheme.startVal
        self.eV = annealingScheme.endVal
        self.aE = annealingScheme.numEpisodes
        self.aK = annealingScheme.kind
        
        self.annealingScheme = annealingScheme

        self.v = self.sV
        self.i = 0
        return

    @abstractmethod
    def __call__(self, Q: np.ndarray, N: Optional[np.ndarray] = None) -> int:
        """ Returns an action given a set of Q-values. N is the number of times each action has been taken. """
        pass

    def anneal(self) -> None:
        """ Decreases the exploration parameter based on the annealing kind. """
        self.i += 1
        {'linear': self._annealLinear, 'exponential': self._annealExponential}[self.aK]()
        return

    def reset(self) -> None:
        """ Resets the exploration parameter and iteration count. """
        self.v = self.sV
        self.i = 0
        return

    def _annealLinear(self) -> None:
        """ Linearly decreases the exploration parameter. """
        self.v = self.eV if self.i > self.aE else self.sV - (self.sV - self.eV) * (self.i / self.aE)
        return
    
    def _annealExponential(self) -> None:
        """ Exponentially decreases the exploration parameter. """
        self.v = self.eV if self.i > self.aE else self.sV * (self.eV / self.sV) ** (self.i / self.aE)
        return


class EpsilonGreedy(ExplorationStrategy):
    """ Epsilon-greedy strategy. """

    @property
    def epsilon(self) -> float:
        """ Probability of taking a random action. """
        return self.v

    def __call__(self, Q: np.ndarray, N: Optional[np.ndarray] = None) -> int:
        """ Returns an action given a set of Q-values. N will be ignored. """
        if np.random.random() < self.epsilon:
            return np.random.randint(Q.shape[0])
        else:
            return np.argmax(Q)
        

class Boltzmann(ExplorationStrategy):
    """ Boltzmann strategy. """

    @property
    def tau(self) -> float:
        """ Temperature of the Boltzmann distribution. """
        return self.v

    def __call__(self, Q: np.ndarray, N: Optional[np.ndarray] = None) -> int:
        """ Returns an action given a set of Q-values. N will be ignored. """
        Q /= self.tau; Q -= np.max(Q)
        return np.random.choice(Q.shape[0], p=np.exp(Q) / np.sum(np.exp(Q)))


class UCB(ExplorationStrategy):
    """ Upper confidence bound strategy. """
    
    @property
    def zeta(self) -> float:
        """ Constant with which lesser explored actions are favored. """
        return self.v
    
    def __call__(self, Q: np.ndarray, N: np.ndarray) -> int:
        """ Returns an action given a set of Q-values and action counts. """
        if 0 in N: return np.where(N == 0)[0][0]
        return np.argmax(Q + np.sqrt(self.zeta * np.log(np.sum(N)) / N))
