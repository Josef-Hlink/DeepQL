""" Statistics utilities. """

import numpy as np


def calculateActionBias(actions: np.ndarray) -> float:
    """ Calculates the percentage one action is preferred over the other. """
    if len(actions.shape) == 3:
        normAction0 =  (actions / np.sum(actions, axis=2, keepdims=True))[:, :, 0]
        actionBiases = np.abs(normAction0 - .5) * 2
        avgActionBiases = np.mean(actionBiases, axis=0)
        return np.mean(avgActionBiases)
    elif len(actions.shape) == 2:
        normAction0 =  (actions / np.sum(actions, axis=1, keepdims=True))[:, 0]
        actionBiases = np.abs(normAction0 - .5) * 2
        return np.mean(actionBiases)
    else:
        raise NotImplementedError
