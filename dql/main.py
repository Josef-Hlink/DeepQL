#!/usr/bin/env python3

import argparse

from dql.agent import DQLAgent, renderEpisodes
from dql.utils.parsewrapper import ParseWrapper
from dql.utils.namespaces import P
from dql.utils.helpers import fixDirectories, PrintIfVerbose
from dql.utils.datamanager import DataManager

import numpy as np
import tensorflow as tf
import gym


def main():
    
    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()
    
    args.seed = args.seed if args.seed is not None else np.random.randint(0, 10**3)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    V, D = args.verbose, args.debug
    printV, printD = PrintIfVerbose(V), PrintIfVerbose(D)
    dataManager = DataManager(P.data, args.runID)

    env = gym.make('CartPole-v1', render_mode='rgb_array')

    agent = DQLAgent(
        explorationStrategy = args.explorationStrategy,
        explorationValue = args.explorationValue,
        alpha = args.alpha,
        gamma = args.gamma,
        annealingRate = args.annealingRate,
        actionSpace = env.action_space.n,
        memorySize = args.memorySize if args.memoryReplay else None,
        batchSize = args.batchSize if args.memoryReplay else None,
        targetUpdateFreq = args.targetFrequency if args.targetNetwork else None,
        stateSpace = env.observation_space.shape[0],
        V = V, D = D
    )

    R = np.empty((args.numRepetitions, args.numEpisodes))
    A = np.empty((args.numRepetitions, args.numEpisodes, env.action_space.n))

    for rep in range(args.numRepetitions):
        printV(f'Running repetition {rep+1} of {args.numRepetitions}')
        results = agent.train(env, args.numEpisodes, V)
        
        R[rep] = results['rewards']
        A[rep] = results['actions']
        
        printD(np.mean(results['rewards']))
        printD(np.sum(results['actions'], axis=0))
        
        dataManager.saveModel(agent.model, rep+1, 'behaviour')
        if agent.usingTN:
            dataManager.saveModel(agent.targetModel, rep+1, 'target')

    dataManager.saveRewards(R)
    dataManager.saveActions(A)
    dataManager.createSummary(data=args)

    if args.render:
        env = gym.make('CartPole-v1', render_mode='human')
        renderEpisodes(env, f'{P.data}/{args.runID}/behaviour_models/{args.numRepetitions-1}.h5', 10, V)

if __name__ == '__main__':
    main()
