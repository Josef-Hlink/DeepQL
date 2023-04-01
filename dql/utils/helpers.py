""" Small helper methods. """

import os
from typing import Iterator, Iterable, Optional
from time import perf_counter
from datetime import datetime

from dql.utils.namespaces import P, UC
from dql.utils.minis import bold, formatRuntime
from dql.utils.progressbar import ProgressBar


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


def prog(iterable: Iterable, verbose: bool, title: Optional[str] = None) -> Iterator:
    """ Adds a progress bar to an iterable and yields its contents. """
    if verbose:
        progressBar = ProgressBar(len(iterable), title)
        for i, item in enumerate(iterable):
            yield item
            progressBar.update(i + 1)
    else:
        title = f' {bold(title)}' if title is not None else ''
        tic = perf_counter()
        print(f'Started{bold(title)} at {datetime.now().strftime("%H:%M:%S")}', end='', flush=True)
        for item in iterable:
            yield item
        print(f'\rFinished{bold(title)} in {formatRuntime(perf_counter() - tic)}' + ' ' * 10)
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
