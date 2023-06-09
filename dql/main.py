#!/usr/bin/env python3

import argparse
from time import perf_counter
from pathlib import Path

from dql.agents.agent import DQLAgent
from dql.agents.annealing import AnnealingScheme, getAnnealingScheme
from dql.agents.exploration import EpsilonGreedy, Boltzmann, UCB

from dql.utils.parsewrapper import ParseWrapper
from dql.utils.namespaces import P
from dql.utils.minis import getRSS
from dql.utils.helpers import fixDirectories, PrintIfDebug, renderEpisodes
from dql.utils.datamanager import DataManager, ConcatDataManager
from dql.utils.statistics import calculateActionBias

import numpy as np
import gym


def main():

    rssStart = getRSS()

    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    V, D = args.verbose, args.debug
    printD = PrintIfDebug(D)
    if args.concat:
        dataManager = ConcatDataManager(args.runID)
    else:
        dataManager = DataManager(args.runID)

    annealingScheme: AnnealingScheme = getAnnealingScheme(args.annealingScheme, args.numEpisodes)
    explorationStrategy = {
        'egreedy': EpsilonGreedy,
        'boltzmann': Boltzmann,
        'ucb': UCB
    }[args.explorationStrategy](annealingScheme)

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    agent = DQLAgent(
        explorationStrategy = explorationStrategy,
        alpha = args.alpha,
        gamma = args.gamma,
        batchSize = args.batchSize,
        replayBufferSize = args.memorySize if args.memoryReplay else None,
        targetUpdateFreq = args.targetFrequency if args.targetNetwork else None,
        actionSpace = env.action_space.n,
        stateSpace = env.observation_space.shape[0]
    )

    R = np.zeros((args.numRepetitions, args.numEpisodes), dtype=np.int16)
    A = np.zeros((args.numRepetitions, args.numEpisodes, env.action_space.n), dtype=np.int16)
    L = []

    tic = perf_counter()

    for rep in range(args.numRepetitions):

        print(f'Running repetition {rep+1} of {args.numRepetitions}')
        agent.randomWarmup(env, args.numWarmupSteps)
        results = agent.train(env, args.numEpisodes, V)
        
        R[rep] = results.rewards
        A[rep] = results.actions
        L.append(results.losses)
        
        printD(f'Average reward: {np.mean(results.rewards):.3f}')
        printD(f'Average action bias: {calculateActionBias(results.actions):.3f}')
        printD(f'Average loss: {np.nanmean(results.losses):.3f}')
        
        dataManager.saveModel(agent.model, kind='behaviour')
        if agent.usingTN:
            dataManager.saveModel(agent.targetModel, kind='target')

        agent.reset()

    toc = perf_counter()
    data = args.copy()
    data['avgRuntime'] = (toc - tic) / args.numRepetitions
    data['annealingScheme'] = annealingScheme.dict()

    dataManager.saveRewards(R)
    dataManager.saveActions(A)
    dataManager.saveLosses(L)
    dataManager.createSummary(data)

    if args.render:
        env = gym.make('CartPole-v1', render_mode='human')
        modelPath = Path(P.data) / args.runID / 'behaviour_models' / f'{args.numRepetitions}.h5'
        renderEpisodes(env, modelPath, 10, V)

    rssEnd = getRSS()

    printD(f'RSS start: {rssStart:.2f} MB')
    printD(f'RSS end: {rssEnd:.2f} MB')
    printD(f'RSS diff: {rssEnd - rssStart:.2f} MB')



if __name__ == '__main__':
    main()
