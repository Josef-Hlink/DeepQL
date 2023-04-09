""" All plot classes are defined here. """

from abc import ABC, abstractmethod

from dql.utils.namespaces import LC

import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8')
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter


class Plot(ABC):

    @abstractmethod
    def __init__(self, data: np.ndarray) -> None:
        assert len(data.shape) in [2, 3], 'data must be 2D or 3D'
        self.data = data
        self.nReps = data.shape[0]
        self.nEps = data.shape[1]
        self.smWindow = (self.nEps // 10 + 1)
        if self.smWindow % 2 == 0:
            self.smWindow += 1
        self.fig = None
        pass

    def getFig(self) -> plt.Figure:
        """ Return the figure. """
        return self.fig

    @staticmethod
    def smooth3(vec: np.ndarray, window: int, minV: int = None, maxV: int = None) -> np.ndarray:
        """
        Smooths a vector using a Savitzky-Golay filter (polynomial order 3).
        If no minV or maxV is given, the min and max of the vector are used.
        """
        assert len(vec.shape) == 1, 'vec must be 1D'
        assert window % 2 == 1, 'window must be odd'
        assert window <= vec.shape[0], 'window must be smaller than vec'
        if minV is not None or maxV is not None:
            return np.clip(savgol_filter(vec, window, 3), minV, maxV)
        else:
            return np.clip(savgol_filter(vec, window, 3), np.min(vec), np.max(vec))
    
    @staticmethod
    def norm(arr: np.ndarray, wrt: np.ndarray = None) -> np.ndarray:
        """ Normalizes an array to the range [0, 1], possibly w.r.t. another array. """
        if wrt is None:
            wrt = arr
        return (arr - np.min(wrt)) / (np.max(wrt) - np.min(wrt))


class ColorPlot(Plot):

    def __init__(self, data: np.ndarray, label: str, title: str) -> None:
        super().__init__(data)
        self.label = label
        self.title = title
        self.cmap = plt.cm.RdYlGn if self.label == 'reward' else plt.cm.RdYlGn_r
        self._processData()
        self._plot()
        return

    def _processData(self) -> None:
        """ Sets all properties containing data to be used in plotting. """
        self.normData = self.norm(self.data)
        self.avgData = np.mean(self.data, axis=0)
        self.minV, self.maxV = (0, 1) if self.label == 'action bias' else (np.min(self.data), np.max(self.data))
        self.smoothed = self.smooth3(self.avgData, self.smWindow, self.minV, self.maxV)
        self.normSmoothed = self.norm(self.smoothed, wrt=self.data)
        return

    def _plot(self) -> None:
        """ Sets up the figure and axes. """
        # instantiate figure with two axes objects
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        # learning curve
        self._addDataTo1()
        self._fmtXAxis1()
        self._fmtYAxis1()
        self._addLegendTo1()
        # heatmap
        self._addDataTo2()
        self._fmtXAxis2()
        self._fmtYAxis2()
        # colorbar
        self._addColorbar()
        self._fmtColorbar()
        # general
        self._fmtGeneral()
        self.fig.suptitle(f'{self.label.title()} {self.title}', fontsize=14, weight='bold')
        return

    def _addDataTo1(self) -> None:
        """ Learning curve; raw and smoothed average (over repetitions). """
        self.ax1.plot(self.avgData, color='k', alpha=.3, zorder=1)
        self.ax1.scatter(
            x = np.arange(self.avgData.shape[0]), y = self.smoothed,
            color = self.cmap(self.normSmoothed), s = 5, zorder = 2
        )
        return

    def _addDataTo2(self) -> None:
        """ Heatmap showing the data of all individual repetitions. """
        self.ax2.imshow(self.normData, aspect='auto', cmap=self.cmap, vmin=0, vmax=1)
        return

    def _fmtYAxis1(self) -> None:
        """ Limits and label. """
        if self.label == 'action bias':
            self.ax1.set_ylim(0, 1)
        else:
            self.ax1.set_ylim(0, 500)
        self.ax1.set_ylabel(f'avg. {self.label}')
        return

    def _fmtYAxis2(self) -> None:
        """ Label, ticks and tick labels (some weird shifting needs to be done to get the labels right). """
        self.ax2.set_ylabel('avg. per run')
        self.ax2.set_yticks(np.arange(-.5, self.nReps-.5))
        yTickLabels = [r'$\mathbf{' + f'{i+1}:' + r'}$' + f'{np.mean(self.data[i]):.2f}' for i in range(self.nReps)]
        self.ax2.yaxis.set_major_formatter(ticker.NullFormatter())
        self.ax2.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(self.nReps)))
        self.ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(yTickLabels))
        return

    def _fmtXAxis1(self) -> None:
        """ No need for this, ax1 shares the x-axis with ax2. """
        return

    def _fmtXAxis2(self) -> None:
        """ Label, ticks and tick labels. """
        self.ax2.set_xlabel('episode', fontsize=12)
        self.ax2.set_xticks(np.linspace(0, self.data.shape[1], 11))
        self.ax2.xaxis.set_major_formatter(lambda x, _: f'{x / 1000:.0f}k' if x >= 1000 else f'{x:.0f}')
        return
    
    def _addLegendTo1(self) -> None:
        """ Legend for the smoothed average showing info on the total average and how much the curve was smoothed. """
        self.ax1.legend(handles = [plt.Line2D(
            [0], [0],
            color = self.cmap(np.mean(self.normSmoothed)),
            linewidth = 3,
            label = f'avg. {np.mean(self.smoothed):.2f}\n(sm. win. {self.smWindow})'
        )])
        return
    
    def _fmtGeneral(self) -> None:
        """ Grid and spines for both axes. """
        for ax in [self.ax1, self.ax2]:
            ax.grid(color='w', linestyle='--', linewidth=1, alpha=.75)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')
                spine.set_linewidth(1.5)
        return

    def _addColorbar(self) -> None:
        """ Colorbar for both the heatmap and the smoothed average "line" plot. """
        self.fig.subplots_adjust(right=.85)
        self.cax = self.fig.add_axes([0.87, 0.15, 0.03, 0.70])
        self.cb = self.fig.colorbar(self.ax2.images[0], cax=self.cax, drawedges=True)
        return

    def _fmtColorbar(self) -> None:
        """ Colorbar ticks and tick labels, and colorbar spines. """
        self.cb.set_ticks(np.linspace(0, 1, 6))
        self.cb.set_ticklabels([f'{t:.1f}' for t in np.linspace(self.minV, self.maxV, 6)])
        self.cb.outline.set_edgecolor('k')
        self.cb.outline.set_linewidth(1.5)
        return


class LossPlot(Plot):

    def __init__(self, data: np.ndarray, title: str) -> None:
        super().__init__(data)
        self.title = title
        self._processData()
        self._plot()
        return

    def _processData(self) -> None:
        """ Smooth the data and get the min and max values. """
        self.avgData = np.mean(self.data, axis=0)
        self.minV, self.maxV = np.nanmin(self.avgData), np.nanmax(self.avgData)
        self.smoothed = self.smooth3(self.avgData, self.smWindow, self.minV, self.maxV)
        return
    
    def _plot(self) -> None:
        """ Plot the data. """
        # instantiate figure with one axes object
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        # loss curve
        self._addData()
        self._fmtYAxis()
        self._fmtXAxis()
        self._fmt()
        self.ax.set_title(f'Loss {self.title}', fontsize=14, fontweight='bold')
        return
    
    def _addData(self) -> None:
        """ Loss curve. """
        self.ax.plot(self.avgData, color='k', alpha=0.3)
        self.ax.plot(self.smoothed, color='k', label=f'avg. {np.nanmean(self.avgData):.2f}\n(sm. win. {self.smWindow})')
        self.ax.legend()
        return
    
    def _fmtYAxis(self) -> None:
        """ Label, log scale, ticks and grid. """
        self.ax.set_ylabel('loss [MSE]')
        self.ax.set_yscale('log')
        self.ax.minorticks_on()
        self.ax.grid(which='minor', alpha=0.5, axis='y')
        return
    
    def _fmtXAxis(self) -> None:
        """ Label, ticks and tick labels. """
        self.ax.set_xlabel('training batch')
        self.ax.xaxis.set_major_formatter(lambda x, _: f'{x / 1000:.1f}k' if x >= 1000 else f'{x:.0f}')
        return
    
    def _fmt(self) -> None:
        """ Title and spines. """
        for spine in self.ax.spines.values():
            spine.set_edgecolor('k')
            spine.set_linewidth(1.5)
        self.fig.tight_layout()
        return


class ComparisonPlot(Plot):
    """
    Plots a given number of different configurations in one figure.
    The upper figure will contain the rewards (avg over reps) with a low opacity,
    as well as a smoothed version of the same color, but with a higher opacity.
    The lower figure works the same way, but for the action biases.
    """

    def __init__(self, data: list[tuple[np.ndarray]], labels: list[str], title: str) -> None:
        """
        The data takes the following form:
        For each entry in the list, a tuple with (rewards, action biases) is expected.
        """
        assert len(data) == len(labels), 'data and labels must have the same length'
        for entry in data:
            assert len(entry) == 2, 'each entry in data must be a tuple with (rewards, action biases)'
        self.data = data
        self.labels = labels
        self.title = title
        self.colors = {
            'BL': 'tab:green', 'ER': 'tab:blue',
            'TN': 'tab:red', 'TR': 'tab:purple',
            'EA1-EG': 'tab:blue', 'EA4-EG': 'tab:cyan',
            'EA1-BM': 'tab:red', 'EA4-BM': 'tab:pink',
            'EA0-UC': 'tab:green'
        }
        self.fullLabels = {
            'BL': 'Baseline', 'ER': 'Experience Replay',
            'TN': 'Target Network', 'TR': 'Target Network + Experience Replay',
            'EA1-EG': f'{LC.e}-greedy (1)', 'EA1-BM': 'Boltzmann (1)',
            'EA4-EG': f'{LC.e}-greedy (4)', 'EA4-BM': 'Boltzmann (4)',
            'EA0-UC': 'UCB (0)'
        }
        self._processData()
        self._plot()
        return
    
    def _processData(self) -> None:
        self.db = {l: {'r': d[0], 'ab': d[1]} for l, d in zip(self.labels, self.data)}
        self.nReps = self.db[self.labels[0]]['r'].shape[0]
        self.nEps = self.db[self.labels[0]]['r'].shape[1]
        self.smWindow = (self.nEps // 10 + 1)
        if self.smWindow % 2 == 0:
            self.smWindow += 1
        return
    
    def _plot(self) -> None:
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for l in self.labels:
            self._plotRewards(l)
            self._plotActionBiases(l)
        self.ax1.legend(loc='upper left')
        self._fmtGeneral()
        self.fig.suptitle(self.title, fontsize=14, weight='bold')
        self.fig.tight_layout()
        return
    
    def _plotRewards(self, label: str) -> None:
        R = self.db[label]['r']
        avg = np.mean(R, axis=0)
        smoothed = self.smooth3(avg, self.smWindow)
        self.ax1.plot(avg, color=self.colors[label], alpha=.2)
        self.ax1.plot(smoothed, linewidth=3, color=self.colors[label], alpha=.8, label=self.fullLabels[label])
        self.ax1.set_ylabel('reward')
        self.ax1.set_xlim(0, self.nEps)
        return
    
    def _plotActionBiases(self, label: str) -> None:
        AB = self.db[label]['ab']
        avg = np.mean(AB, axis=0)
        smoothed = self.smooth3(avg, self.smWindow)
        self.ax2.plot(avg, color=self.colors[label], alpha=.2)
        self.ax2.plot(smoothed, linewidth=3, color=self.colors[label], alpha=.8, label=self.fullLabels[label])
        self.ax2.set_ylabel('action bias')
        self.ax2.set_xlim(0, self.nEps)
        self.ax2.set_xlabel('episode', fontsize=12)
        self.ax2.set_xticks(np.linspace(0, self.nEps, 11))
        self.ax2.xaxis.set_major_formatter(lambda x, _: f'{x / 1000:.0f}k' if x >= 1000 else f'{x:.0f}')
        return
    
    def _fmtGeneral(self) -> None:
        for ax in [self.ax1, self.ax2]:
            ax.grid(color='w', linestyle='--', linewidth=1, alpha=.75)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')
                spine.set_linewidth(1.5)
