""" Houses the DataManager class. """

import json
from pathlib import Path
from time import perf_counter

from dql.utils.namespaces import P
from dql.utils.minis import formatRuntime

import numpy as np
from keras import Sequential


class DataManager:

    def __init__(self, runID: str) -> None:
        self.basePath = Path(P.data) / runID
        if self.basePath.exists():
            i = 1
            while True:
                self.basePath = Path(P.data) / f'{runID}_{i}'
                if not self.basePath.exists():
                    break
                i += 1
        self.basePath.mkdir(parents=True, exist_ok=True)
        self.tic = perf_counter()
        return
    
    def saveModel(self, model: Sequential, repetition: int, name: str) -> None:
        """ Saves a trained model. """
        folder = self.basePath / f'{name}_models'
        folder.mkdir(parents=True, exist_ok=True)
        model.save(folder / f'{repetition}.h5')
        return
    
    def saveRewards(self, rewards: np.ndarray) -> None:
        """ Saves the rewards. """
        self.avgReward = np.mean(rewards)
        np.save(self.basePath / 'rewards.npy', rewards)
        return
    
    def saveActions(self, actions: np.ndarray) -> None:
        """ Saves the actions. """
        normActions = actions / np.sum(actions, axis=2, keepdims=True)
        self.avgActionBias = np.mean(normActions[:, :, 0])
        np.save(self.basePath / 'actions.npy', actions)
        return
    
    def createSummary(self, data: dict) -> None:
        """ Creates a summary of the run. """
        data.pop('verbose')
        data.pop('debug')
        if not data.memoryReplay:
            data.pop('memoryReplay')
            data.pop('memorySize')
            data.pop('batchSize')
        if not data.targetNetwork:
            data.pop('targetNetwork')
            data.pop('targetFrequency')
        data['runtime'] = formatRuntime(perf_counter() - self.tic)
        data['avgReward'] = self.avgReward
        data['avgActionBias'] = self.avgActionBias
        with open(self.basePath / 'summary.json', 'w') as f:
            json.dump(data, f, indent=2)
        return
