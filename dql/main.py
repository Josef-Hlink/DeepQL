#!/usr/bin/env python3

import argparse

from dql.utils.parsewrapper import ParseWrapper
from dql.utils.helpers import fixDirectories, PrintIfVerbose
from dql.agent import BaseAgent, ReplayAgent

import gym


def main():
    
    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()
    
    V, D = args['verbose'], args['debug']
    printV = PrintIfVerbose(V)

    renderMode = 'human' if False else 'none'
    env = gym.make('CartPole-v1', render_mode=renderMode)

    baseAgent = BaseAgent(
        explorationStrategy = args['explorationStrategy'],
        explorationValue = args['explorationValue'],
        alpha = args['alpha'],
        gamma = args['gamma'],
        annealingTemperature = args['annealingTemperature'],
        actionSpace = env.action_space.n,
        stateSpace = env.observation_space.shape[0]
    )

    baseScores = baseAgent.train(env, args['numEpisodes'], env.spec.max_episode_steps, V, D)
    printV(f'Base scores: {baseScores}')


    replayAgent = ReplayAgent(
        explorationStrategy = args['explorationStrategy'],
        explorationValue = args['explorationValue'],
        alpha = args['alpha'],
        gamma = args['gamma'],
        annealingTemperature = args['annealingTemperature'],
        actionSpace = env.action_space.n,
        stateSpace = env.observation_space.shape[0],
        batchSize = args['batchSize'],
        memorySize = args['memorySize']
    )

    replayScores = replayAgent.train(env, args['numEpisodes'], env.spec.max_episode_steps, V, D)
    printV(f'Replay scores: {replayScores}')


if __name__ == '__main__':
    main()
