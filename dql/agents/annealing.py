"""
Annealing schemes for different exploration strategies.
The format is:
    startVal: The starting value of the exploration parameter.
    endVal: The ending value of the exploration parameter.
    window: The fraction of episodes over which to anneal.
    kind: The kind of annealing to use. Is one of 'linear' or 'exponential'.
"""


class AnnealingScheme:
    """ Annealing scheme. """
    startVal: float
    endVal: float
    window: float
    kind: str
    numEpisodes: int

    def __init__(self, numEpisodes: int) -> None:
        """
        Initializes an annealing scheme instance.

        Parameters:
            - [int] `numEpisodes`: Total number of episodes.
                * `self.numEpisodes` will reflect the number of episodes over which to anneal.
        """
        self.numEpisodes = self.window * numEpisodes
        return
    
    def dict(self) -> dict[str, any]:
        """ Returns a dictionary representation of the annealing scheme. """
        return dict(
            id = int(self.__class__.__name__[-1]),
            startVal = self.startVal,
            endVal = self.endVal,
            window = self.window,
            kind = self.kind
        )


def getAnnealingScheme(id: int, numEpisodes: int) -> AnnealingScheme:
    """ Returns an annealing scheme based on its id. """
    return eval(f'AnnealingScheme{id}')(numEpisodes)


class AnnealingScheme0(AnnealingScheme):
    """ No annealing, stays at 1.0. """
    startVal = 1.0
    endVal = 1.0
    window = 0.0
    kind = 'linear'
    numEpisodes = None

class AnnealingScheme1(AnnealingScheme):
    """ Exponentially anneals the exploration parameter from 1.0 to 0.01 over 0.8 of the episodes. """
    startVal = 1.0
    endVal = 0.01
    window = 0.8
    kind = 'exponential'
    numEpisodes = None

class AnnealingScheme2(AnnealingScheme):
    """ Linearly anneals the exploration parameter from 1.0 to 0.01 over 0.8 of the episodes. """
    startVal = 1.0
    endVal = 0.01
    window = 0.8
    kind = 'linear'
    numEpisodes = None

class AnnealingScheme3(AnnealingScheme):
    """ Exponentially anneals the exploration parameter from 1.0 to 0.1 over 0.5 of the episodes. """
    startVal = 1.0
    endVal = 0.1
    window = 0.5
    kind = 'exponential'
    numEpisodes = None

class AnnealingScheme4(AnnealingScheme):
    """ Linearly anneals the exploration parameter from 1.0 to 0.1 over 0.5 of the episodes. """
    startVal = 1.0
    endVal = 0.1
    window = 0.5
    kind = 'linear'
    numEpisodes = None
