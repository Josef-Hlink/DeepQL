import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
from .namespaces import P, UC


class ParseWrapper:
    """ Adds and parses command line arguments for the main script. """
    def __init__(self, parser: ArgumentParser):
        """ Adds arguments to the passed parser object and parses them. """
        parser.add_argument('-e', '--exploration-strategy', dest='explorationStrategy',
            type=str, default='e-greedy', choices=['e-greedy', 'boltzmann', 'ucb'],
            help='Exploration strategy, {e-greedy, boltzmann, ucb}'
        )
        parser.add_argument('-v', '--exploration-value', dest='explorationValue',
            type=float, default=None, help=f'Exploration value (e-greedy: {UC.e}, boltzmann: {UC.t}, ucb: {UC.z})'
        )
        parser.add_argument('-a', '--alpha', dest='alpha',
            type=float, default=0.1, help=f'Learning rate ({UC.a})'
        )
        parser.add_argument('-g', '--gamma', dest='gamma',
            type=float, default=0.99, help=f'Discount factor ({UC.g})'
        )
        parser.add_argument('-n', '--n-episodes', dest='numEpisodes',
            type=int, default=100, help='Budget in episodes'
        )
        parser.add_argument('-r', '--n-reps', dest='numRepetitions',
            type=int, default=5, help='Number of repetitions'
        )
        parser.add_argument('-b', '--batch-size', dest='batchSize',
            type=int, default=32, help='Batch size'
        )
        parser.add_argument('-i', '--run-id', dest='runID',
            type=str, default=None,
            help='Run ID used for saving checkpoints and plots (default: <explStrat>(<explVal>)--<timeStamp>'
        )

        parser.add_argument('-S', '--seed', type=int, default=42, help='Random seed')
        parser.add_argument('-C', '--checkpoints', action='store_true', help='Save checkpoints')
        parser.add_argument('-V', '--verbose', action='store_true', help='Verbose output')
        parser.add_argument('-D', '--debug', action='store_true', help='Debug mode')

        self.defaults = ParseWrapper.resolveDefaultNones(vars(parser.parse_args([])))
        self.args = ParseWrapper.resolveDefaultNones(vars(parser.parse_args()))
        self.validate()
        return

    def __call__(self) -> dict[str, any]:
        """ Returns the parsed and processed arguments as a standard dictionary. """
        if self.args['verbose']:
            print('-' * 80)
            print('Experiment will be ran with the following parameters:')
            for arg, value in self.args.items():
                if self.defaults[arg] != value:
                    print(f'\033[1m{arg:>19}\033[0m | {value}')
                else:
                    print(f'{arg:>19} | {value}')
            print('-' * 80)
        return self.args

    @staticmethod
    def resolveDefaultNones(args: dict[str, any]) -> dict[str, any]:
        """ Resolves default values for exploration value and run ID. """
        resolvedArgs = args.copy()
        defaultExplorationValues = {'e-greedy': 0.1, 'boltzmann': 1.0, 'ucb': 1.0}
        if args['explorationValue'] is None:
            resolvedArgs['explorationValue'] = defaultExplorationValues[args['explorationStrategy']]
        if args['runID'] is None:
            resolvedArgs['runID'] = args['explorationStrategy'] + \
                '(' + str(resolvedArgs['explorationValue']) + ')-' + \
                '-' + datetime.now().strftime('%Y%m%d-%H%M%S')
        return resolvedArgs

    def validate(self) -> None:
        """ Checks the validity of all passed values for the experiment. """
        if self.args['explorationStrategy'] == 'e-greedy':
            assert 0 <= self.args['explorationValue'] <= 1, \
                f'For e-greedy exploration, {UC.e} value must be in [0, 1]'
        elif self.args['explorationStrategy'] == 'boltzmann':
            assert self.args['explorationValue'] > 0, \
                f'For boltzmann exploration, {UC.t} value must be > 0'
        elif self.args['explorationStrategy'] == 'ucb':
            assert self.args['explorationValue'] > 0, \
                f'For ucb exploration, {UC.z} value must be > 0'
        
        assert 0 <= self.args['alpha'] <= 1, \
            f'Learning rate {UC.a} must be in [0, 1]'
        assert 0 <= self.args['gamma'] <= 1, \
            f'Discount factor {UC.g} must be in [0, 1]'
        assert 0 < self.args['numEpisodes'] < 1000, \
            'Number of episodes must be in {0 .. 1000}'
        assert 0 < self.args['numRepetitions'] < 100, \
            'Number of repetitions must be in {0 .. 100}'
        assert self.args['batchSize'] in {1, 2, 4, 8, 16, 32, 64, 128}, \
            'Batch size must be a power of 2 in {1 .. 128}'
        return