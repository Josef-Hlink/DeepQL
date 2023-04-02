#!/usr/bin/env python3

import argparse

from dql.agent import DQLAgent, renderEpisodes
from dql.utils.parsewrapper import ParseWrapper
from dql.utils.namespaces import P
from dql.utils.helpers import fixDirectories, PrintIfDebug
from dql.utils.datamanager import DataManager

import numpy as np
import tensorflow as tf
import gym
import psutil


def main():

    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()
    
    args.seed = args.seed if args.seed is not None else np.random.randint(0, 10**3)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    V, D = args.verbose, args.debug
    printD = PrintIfDebug(D)
    dataManager = DataManager(args.runID)
    memoryStart = psutil.virtual_memory().used / 10**9

    env = gym.make('CartPole-v1', render_mode='rgb_array')

    agent = DQLAgent(
        explorationStrategy = args.explorationStrategy,
        explorationValue = args.explorationValue,
        alpha = args.alpha,
        gamma = args.gamma,
        annealingRate = args.annealingRate,
        actionSpace = env.action_space.n,
        memorySize = args.memorySize if args.memoryReplay else None,
        batchSize = args.batchSize,
        targetUpdateFreq = args.targetFrequency if args.targetNetwork else None,
        stateSpace = env.observation_space.shape[0]
    )

    R = np.empty((args.numRepetitions, args.numEpisodes))
    A = np.empty((args.numRepetitions, args.numEpisodes, env.action_space.n))

    memory = np.empty((args.numRepetitions))

    for rep in range(args.numRepetitions):
        print(f'Running repetition {rep+1} of {args.numRepetitions}')
        results = agent.train(env, args.numEpisodes, V)
        
        R[rep] = results['rewards']
        A[rep] = results['actions']
        
        printD(np.mean(results['rewards']))
        printD(np.sum(results['actions'], axis=0))
        
        dataManager.saveModel(agent.model, rep+1, 'behaviour')
        if agent.usingTN:
            dataManager.saveModel(agent.targetModel, rep+1, 'target')

        memory[rep] = (psutil.virtual_memory().used / 10**9) - memoryStart

    dataManager.saveRewards(R)
    dataManager.saveActions(A)
    dataManager.createSummary(data=args)

    del agent, env, dataManager, R, A

    if args.render:
        env = gym.make('CartPole-v1', render_mode='human')
        renderEpisodes(env, f'{P.data}/{args.runID}/behaviour_models/{args.numRepetitions-1}.h5', 10, V)

    print('memory used after each repetition:')
    print(memory)


if __name__ == '__main__':
    main()
