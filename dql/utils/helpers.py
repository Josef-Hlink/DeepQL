""" Small helper methods. """

import os
from datetime import datetime

from dql.utils.namespaces import P, UC
from dql.utils.minis import bold

import numpy as np
from keras.models import load_model
from gym import Env


def fixDirectories() -> None:
    """ Creates the necessary directories if they don't exist. """
    for attr, value in P.__dict__.items():
        if attr.startswith('_'):
            continue
        dirName, path = attr, value
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'created {bold(dirName)} directory at `{path}`')
    return


def renderEpisodes(env: Env, modelPath: str, numEpisodes: int = 10, V: bool = False) -> None:
    """ Renders a number of episodes using a saved model. """

    assert env.render_mode == 'human', 'Environment must be rendered in human mode.'
    model = load_model(modelPath)
    
    for _ in range(numEpisodes):
        state, _ = env.reset()
        done, truncated = False, False
        steps = 0
        
        while not done and not truncated:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
            steps += 1
            state, _, done, truncated, _ = env.step(action)
            env.render()
        
        if V: print(f'Episode finished after {steps} steps.')
    
    env.close()
    return


class PrintIfVerbose:
    """ Prints only if verbose is True. """
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class PrintIfDebug:
    """ Prints only if debug is True. """
    def __init__(self, debug: bool):
        self.debug = debug

    def __call__(self, *args, **kwargs):
        if self.debug:
            debugTag = ' \033[1m\033[96mDEBUG\033[0m '
            timeStamp = ' ' + datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' '
            print(UC.tl + UC.hd * 28 + debugTag + UC.hd + timeStamp + UC.hd * 28 + UC.tr)
            print(' ', end='', flush=True)
            print(*args, **kwargs)
            print(UC.bl + UC.hd * 78 + UC.br)
