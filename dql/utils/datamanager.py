""" Houses the DataManager class. """

import json
from pathlib import Path

from dql.utils.namespaces import P
from dql.utils.minis import formatRuntime, DotDict

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
        biases = np.empty(actions.shape[0])
        for i in range(actions.shape[0]):
            bias = np.mean(normActions[i, :, 0])
            biases[i] = 1 - bias if bias < .5 else bias
        self.avgActionBias = np.mean(biases)
        np.save(self.basePath / 'actions.npy', actions)
        return
    
    def saveLosses(self, losses: list[np.ndarray]) -> None:
        """ Saves the losses. """
        maxLen = max([len(l) for l in losses])
        for i in range(len(losses)):
            if len(losses[i]) < maxLen:
                losses[i] = np.pad(losses[i], (0, maxLen - len(losses[i])), 'constant', constant_values=np.nan)
        losses = np.array(losses)
        self.avgLoss = float(np.nanmean(losses))
        np.save(self.basePath / 'losses.npy', losses)
        return
    
    def createSummary(self, data: DotDict) -> None:
        """ Creates a summary of the run. """
        summary = dict(
            meta = dict(
                runID = data.runID,
                numRepetitions = data.numRepetitions,
                numEpisodes = data.numEpisodes,
                seed = data.seed
            ),
            params = dict(
                explorationStrategy = data.explorationStrategy,
                annealingScheme = data.annealingScheme,
                experienceReplay = data.experienceReplay,
                targetNetwork = data.targetNetwork,
                replayBufferSize = data.replayBufferSize if data.experienceReplay else None,
                targetFrequency = data.targetFrequency if data.targetNetwork else None,
                explorationValue = data.explorationValue,
                alpha = data.alpha,
                gamma = data.gamma,
                batchSize = data.batchSize
            ),
            results = dict(
                avgRuntime = formatRuntime(data.avgRuntime),
                avgReward = self.avgReward,
                avgActionBias = self.avgActionBias,
                avgLoss = self.avgLoss
            )
        )        
        with open(self.basePath / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return
