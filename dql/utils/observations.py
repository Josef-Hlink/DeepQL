""" Houses the Observation and ObservationSet types. """

from typing import Optional, Iterable

from numpy import array, ndarray


class Observation:
    """ Type to easily pass around an observation made by an agent. """

    def __init__(self, s: ndarray, a: int, r: float, s_: ndarray, done: bool):
        self._s = s
        self._a = a
        self._r = r
        self._s_ = s_
        self._done = done
        return
    
    def unpack(self) -> tuple[ndarray, int, float, ndarray, bool]:
        """
        Unpacks the observation into its respective components.
        
        Returns:
            - state
            - action
            - reward
            - next state
            - done flag
        """
        return self._s, self._a, self._r, self._s_, self._done

    @property
    def s(self) -> ndarray:
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
    def s_(self) -> ndarray:
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
        self.observations = observations if observations is not None else []
        return
    
    def add(self, observation: Observation) -> None:
        """ Adds an observation to the set. """
        self.observations.append(observation)
        return

    def unpack(self) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
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
            array([o.s for o in self.observations]),
            array([o.a for o in self.observations]),
            array([o.r for o in self.observations]),
            array([o.s_ for o in self.observations]),
            array([o.done for o in self.observations])
        )

    @property
    def totalReward(self) -> float:
        """ The total reward of the observations in the set. """
        return sum(o.r for o in self.observations)

    @property
    def nLeft(self) -> int:
        """ The number of left actions the agent took in the set. """
        return sum(o.a == 0 for o in self.observations)

    @property
    def nRight(self) -> int:
        """ The number of right actions the agent took in the set. """
        return sum(o.a == 1 for o in self.observations)

    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, index: int) -> Observation:
        return self.observations[index]
    
    def __iter__(self):
        return iter(self.observations)
