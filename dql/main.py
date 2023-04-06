#!/usr/bin/env python3

import argparse
from time import perf_counter
from pathlib import Path

from dql.agents.agent import DQLAgent
from dql.agents.annealing import AnnealingScheme, getAnnealingScheme
from dql.agents.exploration import EpsilonGreedy, Boltzmann, UCB

from dql.utils.parsewrapper import ParseWrapper
from dql.utils.namespaces import P
from dql.utils.helpers import getMemoryUsage, fixDirectories, PrintIfDebug, renderEpisodes
from dql.utils.datamanager import DataManager

import numpy as np
import tensorflow as tf
import gym

def main():

    rssStart = getMemoryUsage()

    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()
    
    args.seed = args.seed if args.seed is not None else np.random.randint(0, 10**3)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    V, D = args.verbose, args.debug
    printD = PrintIfDebug(D)
    dataManager = DataManager(args.runID)

    # hacky solution to get around importing all the annealing schemes or 
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
        results = agent.train(env, args.numEpisodes, V)
        
        R[rep] = results['rewards']
        A[rep] = results['actions']
        L.append(results['losses'])
        
        printD(f'Average reward: {np.mean(results["rewards"])}')
        printD(f'Action distribution:\n{np.sum(results["actions"], axis=0) / np.sum(results["actions"])}')
        printD(f'Average loss: {np.mean(results["losses"])}')
        
        dataManager.saveModel(agent.model, rep+1, 'behaviour')
        if agent.usingTN:
            dataManager.saveModel(agent.targetModel, rep+1, 'target')

        agent.reset()

    toc = perf_counter()
    data = args.copy()
    data['avgRuntime'] = (toc - tic) / args.numRepetitions
    data['annealingScheme'] = annealingScheme.dict()

    dataManager.saveRewards(R)
    dataManager.saveActions(A)
    dataManager.saveLosses(L)
    dataManager.createSummary(data)

    del agent, env, dataManager, R, A

    if args.render:
        env = gym.make('CartPole-v1', render_mode='human')
        modelPath = Path(P.data) / args.runID / 'behaviour_models' / f'{args.numRepetitions}.h5'
        renderEpisodes(env, modelPath, 10, V)

    rssEnd = getMemoryUsage()

    printD(f'RSS start: {rssStart:.2f} MB')
    printD(f'RSS end: {rssEnd:.2f} MB')
    printD(f'RSS diff: {rssEnd - rssStart:.2f} MB')



if __name__ == '__main__':
    main()
