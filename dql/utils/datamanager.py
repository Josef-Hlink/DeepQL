""" Houses the DataManager classes. """

import json
from pathlib import Path

from dql.utils.namespaces import P
from dql.utils.minis import formatRuntime, formattedRuntimeToSeconds, DotDict

import numpy as np
from keras.models import Sequential, load_model


class DataManager:

    def __init__(self, runID: str, r: bool = False) -> None:
        """
        When specifying r=True, the DataManager will load the data from the runID folder
        and not try to make a new one when this folder already exists.
        """
        self.basePath = Path(P.data) / runID
        if self.basePath.exists() and not r:
            i = 1
            while True:
                self.basePath = Path(P.data) / f'{runID}_{i}'
                if not self.basePath.exists():
                    break
                i += 1
        self.basePath.mkdir(parents=True, exist_ok=True)
        return
    
    def saveModel(self, model: Sequential, kind: str) -> None:
        """ Saves a trained model. """
        folder = self.basePath / f'{kind}_models'
        folder.mkdir(parents=True, exist_ok=True)
        repetitions = [int(f.stem) for f in folder.iterdir() if f.is_file()]
        repetition = max(repetitions) + 1 if repetitions else 1
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
            meta = self._extractMetaData(data),
            params = self._extractParamData(data),
            results = self._extractResultData(data)
        )
        with open(self.basePath / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return
    
    def _extractMetaData(self, data: DotDict) -> DotDict:
        """ Extracts the meta data from the data. """
        return DotDict(
            runID = data.runID,
            numRepetitions = data.numRepetitions,
            numEpisodes = data.numEpisodes,
            numWarmupSteps = data.numWarmupSteps
        )
    
    def _extractParamData(self, data: DotDict) -> DotDict:
        """ Extracts the parameter data from the data. """
        return DotDict(
            explorationStrategy = data.explorationStrategy,
            annealingScheme = data.annealingScheme,
            experienceReplay = data.experienceReplay,
            targetNetwork = data.targetNetwork,
            replayBufferSize = data.replayBufferSize if data.experienceReplay else None,
            targetFrequency = data.targetFrequency if data.targetNetwork else None,
            alpha = data.alpha,
            gamma = data.gamma,
            batchSize = data.batchSize
        )
    
    def _extractResultData(self, data: DotDict) -> DotDict:
        """ Extracts the result data from the data. """
        return DotDict(
            avgRuntime = formatRuntime(data.avgRuntime),
            avgReward = self.avgReward,
            avgActionBias = self.avgActionBias,
            avgLoss = self.avgLoss
        )

    def loadModel(self, kind: str, repetition: int) -> Sequential:
        """ Loads a trained model. """
        folder = self.basePath / f'{kind}_models'
        repetitions = [int(f.stem) for f in folder.iterdir() if f.is_file()]
        assert repetition in repetitions, f'Repetition {repetition} does not exist in {folder}.'
        return load_model(folder / f'{repetition}.h5')
    
    def loadRewards(self) -> np.ndarray:
        """ Loads the rewards. """
        return np.load(self.basePath / 'rewards.npy')
    
    def loadActions(self) -> np.ndarray:
        """ Loads the actions. """
        return np.load(self.basePath / 'actions.npy')
    
    def loadLosses(self) -> np.ndarray:
        """ Loads the losses. """
        return np.load(self.basePath / 'losses.npy')
    
    def loadSummary(self) -> DotDict:
        """ Loads the summary. """
        with open(self.basePath / 'summary.json', 'r') as f:
            summary = DotDict(json.load(f))
        summary.meta = DotDict(summary.meta)
        summary.params = DotDict(summary.params)
        summary.results = DotDict(summary.results)
        return summary


class ConcatDataManager(DataManager):

    def __init__(self, runID: str) -> None:
        self.basePath = Path(P.data) / runID
        self.basePath.mkdir(parents=True, exist_ok=True)
        return
    
    def saveModel(self, model: Sequential, kind: str) -> None:
        return super().saveModel(model, kind)

    def saveRewards(self, rewards: np.ndarray) -> None:
        """ Saves the rewards. """
        self.avgReward = np.mean(rewards)
        self._saveArr(rewards, 'rewards')
        return
    
    def saveActions(self, actions: np.ndarray) -> None:
        """ Saves the actions. """
        normActions = actions / np.sum(actions, axis=2, keepdims=True)
        biases = np.empty(actions.shape[0])
        for i in range(actions.shape[0]):
            bias = np.mean(normActions[i, :, 0])
            biases[i] = 1 - bias if bias < .5 else bias
        self.avgActionBias = np.mean(biases)
        self._saveArr(actions, 'actions')
        return
    
    def saveLosses(self, losses: list[np.ndarray]) -> None:
        """ Saves the losses. """
        maxLen = max([len(l) for l in losses])
        for i in range(len(losses)):
            if len(losses[i]) < maxLen:
                losses[i] = np.pad(losses[i], (0, maxLen - len(losses[i])), 'constant', constant_values=np.nan)
        losses = np.array(losses)
        self.avgLoss = np.nanmean(losses)
        self._saveArr(losses, 'losses')
        return

    def _saveArr(self, arr: np.ndarray, name: str) -> None:
        """ Appends an array to a .npz file. """
        path = self.basePath / f'{name}.npz'
        if not path.exists():
            np.savez(path, arr_0=arr)
        else:
            with np.load(path) as data:
                latest = int(data.files[-1][-1])
                newData = dict(data)
            newData[f'arr_{latest + 1}'] = arr
            np.savez(path, **newData)
        return

    def createSummary(self, data: DotDict) -> None:
        """ Creates a summary of the run. """
        summary = dict(
            meta = self._extractMetaData(data),
            params = self._extractParamData(data),
            results = self._extractResultData(data)
        )
        with open(self.basePath / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return
    
    def _extractMetaData(self, data: DotDict) -> DotDict:
        """ Extracts the meta data from the data. """
        path = self.basePath / 'summary.json'
        if not path.exists():
            return DotDict(
                runID = data.runID,
                numRepetitions = data.numRepetitions,
                numEpisodes = data.numEpisodes,
                numWarmupSteps = data.numWarmupSteps,
                runs = 1
            )
        with open(path) as f:
            summary = DotDict(json.load(f))
        runs = summary.meta['runs'] + 1
        return DotDict(
            runID = data.runID,
            numRepetitions = data.numRepetitions * runs,
            numEpisodes = data.numEpisodes,
            runs = runs
        )

    def _extractParamData(self, data: DotDict) -> DotDict:
        """ Extracts the parameter data from the data. """
        return super()._extractParamData(data)

    def _extractResultData(self, data: DotDict) -> DotDict:
        """ Extracts the result data from the data. """
        path = self.basePath / 'summary.json'
        if not path.exists():
            return DotDict(
                avgRuntime = formatRuntime(data.avgRuntime),
                avgReward = self.avgReward,
                avgActionBias = self.avgActionBias,
                avgLoss = float(self.avgLoss)
            )
        with open(path) as f:
            summary = DotDict(json.load(f))
        runs = summary.meta['runs'] + 1
        results = DotDict(summary.results)
        return DotDict(
            avgRuntime = formatRuntime((formattedRuntimeToSeconds(results.avgRuntime) * (runs - 1) + data.avgRuntime) / runs),
            avgReward = (results.avgReward * (runs - 1) + self.avgReward) / runs,
            avgActionBias = (results.avgActionBias * (runs - 1) + self.avgActionBias) / runs,
            avgLoss = (results.avgLoss * (runs - 1) + self.avgLoss) / runs
        )

    def loadModel(self, kind: str, repetition: int) -> Sequential:
        """ Loads a trained model. """
        return super().loadModel(kind, repetition)
    
    def loadRewards(self) -> np.ndarray:
        """ Loads the rewards. """
        return self._loadArr('rewards')
    
    def loadActions(self) -> np.ndarray:
        """ Loads the actions. """
        return self._loadArr('actions')
    
    def loadLosses(self) -> np.ndarray:
        """ Loads the losses. """
        path = self.basePath / 'losses.npz'
        # because of the inconsistent length of the losses, we need to load them manually
        with np.load(path) as data:
            individualLosses = []
            for i in range(len(data.files)):
                for j in range(data[f'arr_{i}'].shape[0]):
                    individualLosses.append(data[f'arr_{i}'][j])
        maxLen = max([len(l) for l in individualLosses])
        for i, losses in enumerate(individualLosses):
            if len(losses) < maxLen:
                individualLosses[i] = np.pad(losses, (0, maxLen - len(losses)), 'constant', constant_values=np.nan)
        return np.array(individualLosses)

    def _loadArr(self, name: str) -> np.ndarray:
        """ Loads an array from a .npz file. """
        path = self.basePath / f'{name}.npz'
        with np.load(path) as data:
            arr = np.concatenate([data[f'arr_{i}'] for i in range(len(data.files))], axis=0)
        return arr

    def loadSummary(self) -> DotDict:
        """ Loads the summary of the run. """
        return super().loadSummary()
