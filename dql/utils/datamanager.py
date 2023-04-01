""" Houses the DataManager class. """

import json
from pathlib import Path

import numpy as np
import pandas as pd
from keras import Sequential


class DataManager:

    def __init__(self, dataPath: str, runID: str) -> None:
        self.basePath = Path(dataPath) / runID
        if self.basePath.exists():
            i = 1
            while True:
                self.basePath = Path(dataPath) / f'{runID}_{i}'
                if not self.basePath.exists():
                    break
                i += 1
        self.basePath.mkdir(parents=True, exist_ok=True)
        return
    
    def saveRewards(self, rewards: np.ndarray) -> None:
        """ Saves the rewards. """
        avgR, stdR = np.mean(rewards, axis=0), np.std(rewards, axis=0)
        maxR, minR = np.amax(rewards, axis=0), np.amin(rewards, axis=0)
        df = pd.DataFrame({'avg': avgR, 'std': stdR, 'max': maxR, 'min': minR})
        df.index.name = 'episode'
        df.to_csv(self.basePath / 'rewards.csv')
        np.save(self.basePath / 'rewards.npy', rewards)
        return
    
    def saveActions(self, actions: np.ndarray) -> None:
        """ Saves the actions. """
        normActions = actions / np.sum(actions, axis=2, keepdims=True)
        normActions = np.abs(normActions[:, :, 0] - .5)
        avgA, stdA = np.mean(normActions, axis=0), np.std(normActions, axis=0)
        maxA, minA = np.amax(normActions, axis=0), np.amin(normActions, axis=0)
        df = pd.DataFrame({'avg': avgA, 'std': stdA, 'max': maxA, 'min': minA})
        df.index.name = 'episode'
        df.to_csv(self.basePath / 'actions.csv')
        np.save(self.basePath / 'actions.npy', actions)
        return
    
    def saveModel(self, model: Sequential, repetition: int, name: str) -> None:
        """ Saves a trained model. """
        folder = self.basePath / f'{name}_models'
        folder.mkdir(parents=True, exist_ok=True)
        model.save(folder / f'{repetition}.h5')
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
        with open(self.basePath / 'summary.json', 'w') as f:
            json.dump(data, f, indent=2)
        return
