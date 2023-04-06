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
            type=str, default='egreedy', choices=['egreedy', 'boltzmann', 'ucb'],
            help='Exploration strategy, {egreedy, boltzmann, ucb}'
        )
        parser.add_argument('-as', dest='annealingScheme',
            type=int, default=None, choices=[0, 1, 2, 3, 4],
            help='Annealing scheme, {0, 1, 2, 3, 4}'
        )
        parser.add_argument('-a', dest='alpha',
            type=float, default=0.001, help=f'Learning rate ({UC.a})'
        )
        parser.add_argument('-g', dest='gamma',
            type=float, default=0.999, help=f'Discount factor ({UC.g})'
        )
        parser.add_argument('-ne', dest='numEpisodes',
            type=int, default=1000, help='Budget in episodes'
        )
        parser.add_argument('-nr', dest='numRepetitions',
            type=int, default=5, help='Number of repetitions'
        )
        parser.add_argument('-bs', dest='batchSize',
            type=int, default=32, help='Batch size'
        )
        parser.add_argument('-ER', dest='experienceReplay',
            action='store_true', help='Use experience replay'
        )
        parser.add_argument('-rb', dest='replayBufferSize',
            type=int, default=2000, help='Number of time steps to store in replay buffer'
        )
        parser.add_argument('-TN', dest='targetNetwork',
            action='store_true', help='Use target network'
        )
        parser.add_argument('-tf', dest='targetFrequency',
            type=int, default=100, help='Number of episodes after which to update target network'
        )
        
        parser.add_argument('-I', dest='runID',
            type=str, default=None,
            help='Run ID used for saving checkpoints and plots (default: yyyymmdd-hhmmss)'
        )
        parser.add_argument('-S', '--seed', type=int, default=None, help='Random seed')
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
        defaultAnnealingSchemes = {'egreedy': 1, 'boltzmann': 1, 'ucb': 0}
        if args['annealingScheme'] is None:
            resolvedArgs['annealingScheme'] = defaultAnnealingSchemes[args['explorationStrategy']]
        if args['runID'] is None:
            resolvedArgs['runID'] = datetime.now().strftime('%Y%m%d-%H%M%S')
        return DotDict(resolvedArgs)

    def validate(self) -> None:
        """ Checks the validity of all passed values for the experiment. """
        assert 0 <= self.args.alpha <= 1, \
            f'Learning rate {UC.a} must be in [0, 1]'
        assert 0 <= self.args.gamma <= 1, \
            f'Discount factor {UC.g} must be in [0, 1]'
        assert 0 < self.args.numEpisodes <= 5000, \
            'Number of episodes must be in {1 .. 5000}'
        assert 0 < self.args.numRepetitions <= 100, \
            'Number of repetitions must be in {1 .. 100}'
        assert self.args.batchSize in {1, 2, 4, 8, 16, 32, 64, 128, 256}, \
            'Batch size must be a power of 2 in {1 .. 256}'
        if self.args.memoryReplay:
            assert 0 <= self.args.replayBufferSize <= 20_000, \
                'Memory size must be in {0 .. 20_000}'
            assert self.args.replayBufferSize >= self.args.batchSize, \
                'Replay buffer size must be larger than or equal to batch size'
        if self.args.targetNetwork:
            assert 0 < self.args.targetFrequency <= 1000, \
                'Target network update frequency must be in {1 .. 1000}'
            assert self.args.targetFrequency <= self.args.numEpisodes, \
                'Target network update frequency must be smaller than or equal to number of episodes'
        return
