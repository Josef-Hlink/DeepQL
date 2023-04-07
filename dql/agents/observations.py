""" Houses the Observation and ObservationSet types. """

from typing import Optional, Iterable

import numpy as np


class Observation:
    """ Type to easily pass around an observation made by an agent. """

    def __init__(self, s: np.ndarray, a: int, r: float, s_: np.ndarray, done: bool):
        self._s = s
        self._a = a
        self._r = r
        self._s_ = s_
        self._done = done
        return
    
    def unpack(self) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
        """
        Unpacks the observation into its respective components.
        
        Returns:
            - state
            - action
            - reward
            - next state
            - done flag
        """
        return self.s, self.a, self.r, self.s_, self.done

    @property
    def s(self) -> np.ndarray:
        """ state """
        return self._s
    
    @property
    def a(self) -> int:
        """ action """
        return self._a
    
    @property
    def r(self) -> float:
        """ reward """
        return self._r
    
    @property
    def s_(self) -> np.ndarray:
        """ next state """
        return self._s_
    
    @property
    def done(self) -> bool:
        """ whether s_ was a terminal state """
        return self._done


class ObservationSet:
    """ Type to easily pass around a set of observations made by an agent. """

    def __init__(self, observations: Optional[Iterable[Observation]] = None):
        """ Initializes the set. If no observations are provided, an empty list is used. """
        
        self.observations = list(observations) if observations is not None else []
        return
    
    def add(self, observation: Observation) -> None:
        """ Adds an observation to the set. """
        self.observations.append(observation)
        return

    def unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpacks the observations into their respective arrays.
        
        Returns:
            - states
            - actions
            - rewards
            - next states
            - done flags
        """
        return (
            np.array([o.s for o in self.observations]),
            np.array([o.a for o in self.observations]),
            np.array([o.r for o in self.observations]),
            np.array([o.s_ for o in self.observations]),
            np.array([o.done for o in self.observations])
        )

    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, index: int) -> Observation:
        return ObservationSet(self.observations[index])
    
    def __iter__(self):
        return iter(self.observations)
    
    def __add__(self, other: 'ObservationSet') -> 'ObservationSet':
        return ObservationSet(self.observations + other.observations)

    def __delitem__(self, index: int) -> None:
        del self.observations[index]
        return

class ObservationQueue(ObservationSet):
    """ Type to easily pass around a queue of observations made by an agent. """

    def __init__(self, observations: Optional[Iterable[Observation]] = None, maxSize: Optional[int] = None):
        """ Initializes the queue. If no observations are provided, an empty list is used. """
        super().__init__(observations)
        self.maxSize = maxSize
        return

    def add(self, observation: Observation) -> None:
        """ Adds an observation to the queue. """
        if len(self.observations) == self.maxSize:
            self.observations.pop(0)
        self.observations.append(observation)
        return

    def sample(self, size: int) -> 'ObservationQueue':
        """ Samples a random subset of the queue. """
        return ObservationQueue(np.random.choice(self.observations, size=size))

    def __add__(self, other: 'ObservationQueue') -> 'ObservationQueue':
        return ObservationQueue(self.observations + other.observations)
