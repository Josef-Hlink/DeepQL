""" Houses the ParseWrapper class. """

from argparse import ArgumentParser
from datetime import datetime

from dql.utils.namespaces import UC
from dql.utils.minis import bold, DotDict


class ParseWrapper:
    """ Adds and parses command line arguments for the main script. """
    def __init__(self, parser: ArgumentParser):
        """ Adds arguments to the passed parser object and parses them. """
        parser.add_argument('-es', dest='explorationStrategy',
            type=str, default='e-greedy', choices=['e-greedy', 'boltzmann', 'ucb'],
            help='Exploration strategy, {e-greedy, boltzmann, ucb}'
        )
        parser.add_argument('-ev', dest='explorationValue',
            type=float, default=None, help=f'Exploration value (e-greedy: {UC.e}, boltzmann: {UC.t}, ucb: {UC.z})'
        )
        parser.add_argument('-ar', dest='annealingRate',
            type=float, default=0.999, help=f'Annealing rate (only applicable to {UC.e} and {UC.t})')
        parser.add_argument('-a', dest='alpha',
            type=float, default=0.1, help=f'Learning rate ({UC.a})'
        )
        parser.add_argument('-g', dest='gamma',
            type=float, default=0.99, help=f'Discount factor ({UC.g})'
        )
        parser.add_argument('-ne', dest='numEpisodes',
            type=int, default=1000, help='Budget in episodes'
        )
        parser.add_argument('-nr', dest='numRepetitions',
            type=int, default=5, help='Number of repetitions'
        )
        parser.add_argument('-MR', dest='memoryReplay',
            action='store_true', help='Use memory replay')
        parser.add_argument('-b', dest='batchSize',
            type=int, default=32, help='Batch size'
        )
        parser.add_argument('-m', dest='memorySize',
            type=int, default=2000, help='Memory size'
        )
        parser.add_argument('-TN', dest='targetNetwork',
            action='store_true', help='Use target network')
        parser.add_argument('-f', dest='targetFrequency',
            type=int, default=100, help='Target network update frequency'
        )
        
        parser.add_argument('-I', dest='runID',
            type=str, default=None,
            help='Run ID used for saving checkpoints and plots (default: yyyymmdd-hhmmss)'
        )
        parser.add_argument('-S', '--seed', type=int, default=42, help='Random seed')
        parser.add_argument('-V', '--verbose', action='store_true', help='Verbose output')
        parser.add_argument('-D', '--debug', action='store_true', help='Debug mode')
        parser.add_argument('-R', '--render', action='store_true', help='Render 10 episodes after running')

        self.defaults = ParseWrapper.resolveDefaultNones(vars(parser.parse_args([])))
        self.args = ParseWrapper.resolveDefaultNones(vars(parser.parse_args()))
        self.validate()
        return

    def __call__(self) -> dict[str, any]:
        """ Returns the parsed and processed arguments as a standard dictionary. """
        if self.args.verbose:
            print(UC.hd * 80)
            print('Experiment will be ran with the following parameters:')
            for arg, value in self.args.items():
                if self.defaults[arg] != value:
                    print(f'{bold(arg):>28} {UC.vd} {value}')
                else:
                    print(f'{arg:>20} {UC.vd} {value}')
            print(UC.hd * 80)
        return self.args

    @staticmethod
    def resolveDefaultNones(args: dict[str, any]) -> DotDict[str, any]:
        """ Resolves default values for exploration value and run ID. """
        resolvedArgs = args.copy()
        defaultExplorationValues = {'e-greedy': 0.1, 'boltzmann': 1.0, 'ucb': 2.0}
        if args['explorationValue'] is None:
            resolvedArgs['explorationValue'] = defaultExplorationValues[args['explorationStrategy']]
        if args['runID'] is None:
            resolvedArgs['runID'] = datetime.now().strftime('%Y%m%d-%H%M%S')
        return DotDict(resolvedArgs)

    def validate(self) -> None:
        """ Checks the validity of all passed values for the experiment. """
        if self.args.explorationStrategy == 'e-greedy':
            assert 0 <= self.args.explorationValue <= 1, \
                f'For e-greedy exploration, {UC.e} value must be in [0, 1]'
        elif self.args.explorationStrategy == 'boltzmann':
            assert self.args.explorationValue > 0, \
                f'For boltzmann exploration, {UC.t} value must be > 0'
        elif self.args.explorationStrategy == 'ucb':
            assert self.args.explorationValue > 0, \
                f'For ucb exploration, {UC.z} value must be > 0'
        assert 0 < self.args.annealingRate <= 1, \
            f'Annealing rate must be in (0, 1]'
        
        assert 0 <= self.args.alpha <= 1, \
            f'Learning rate {UC.a} must be in [0, 1]'
        assert 0 <= self.args.gamma <= 1, \
            f'Discount factor {UC.g} must be in [0, 1]'
        assert 0 < self.args.numEpisodes <= 2000, \
            'Number of episodes must be in {1 .. 2000}'
        assert 0 < self.args.numRepetitions <= 100, \
            'Number of repetitions must be in {1 .. 100}'
        if self.args.memoryReplay:
            assert self.args.batchSize in {1, 2, 4, 8, 16, 32, 64, 128}, \
                'Batch size must be a power of 2 in {1 .. 128}'
            assert self.args.batchSize <= self.args.memorySize, \
                'Batch size must be smaller than or equal to memory size'
            assert 0 <= self.args.memorySize <= 20_000, \
                'Memory size must be in {0 .. 20_000}'
        if self.args.targetNetwork:
            assert 0 < self.args.targetFrequency <= 1000, \
                'Target network update frequency must be in {1 .. 1000}'
            assert self.args.targetFrequency <= self.args.numEpisodes, \
                'Target network update frequency must be smaller than or equal to number of episodes'
        return
