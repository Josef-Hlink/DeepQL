from typing import Optional
from time import perf_counter
from datetime import datetime

from .minis import bold, formatRuntime


class ProgressBar:
    w: int = 50
    guard: str = '|'
    block = '\u25a0'  # ■
    empty = '\u25a1'  # □

    def __init__(self, total: int, title: Optional[str] = None):
        self.total = total
        self.title = title
        self.tic = perf_counter()
        print(f'Starting {bold(self.title)} at {datetime.now().strftime("%H:%M:%S")}')
        self.bar = self.guard + self.empty * self.w + self.guard
        self.update(0)
    
    def update(self, i: int):
        """ Updates the progress bar. """
        self.i = i
        self._updatePercentage()
        self._updateBar()
        print('\r' + self.bar, end='', flush=True)
        if self.i == self.total:
            print(f'\r{self.bar[:-5]} - finished in {formatRuntime(perf_counter() - self.tic)}')
    
    def _updatePercentage(self):
        """ Updates the percentage. """
        self.percentage = int(100 * self.i / self.total)
    
    def _updateBar(self):
        """ Updates the progress bar. """
        n = int(self.w * self.i / self.total)
        self.bar = self.guard + self.block * n + self.empty * (self.w - n) + self.guard + f' {self.percentage}%'
