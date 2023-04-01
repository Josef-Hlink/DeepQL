""" Houses the DQLAgent class. """

from collections import deque
from functools import partial

from dql.utils.helpers import PrintIfVerbose, PrintIfDebug, prog
from dql.utils.observations import Observation, ObservationSet

import numpy as np
import gym
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class DQLAgent:
    """
    A Deep Q-Learning agent that is capable of memory replay and double DQN (using a target network).
    """

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingRate: float,
        memorySize: int = None, batchSize: int = None, targetUpdateFreq: int = None,
        actionSpace: int = 2, stateSpace: int = 4, V: bool = False, D: bool = False
    ) -> None:
        """
        Initializes a Deep Q-Learning agent.
        
        Params:
            - [str]   `explorationStrategy`: one of 'e-greedy', 'boltzmann', 'ucb'
            - [float] `explorationValue`: (initial) value of the exploration parameter, e.g. epsilon for e-greedy
            - [float] `alpha`: learning rate for the neural network
            - [float] `gamma`: discount factor for future rewards
            - [float] `annealingRate`: rate at which the exploration parameter is annealed
            - [int]   `memorySize`: size of the replay memory
                * if specified, the agent will use memory replay (you will also need to specify `batchSize`)
            - [int]   `batchSize`: size of the batches used for training in replay phase
            - [int]   `targetUpdateFreq`: number of episodes after which the target network is updated
                * if specified, the agent will be a double DQN agent
            - [int]   `actionSpace`: number of possible actions in the environment
            - [int]   `stateSpace`: number of parameters that represent a state in the environment
            - [bool]  `V`: whether to print verbose output
            - [bool]  `D`: whether to print debug output
        """

        self.getAction: function = {
            'e-greedy': self._epsilonGreedyAction,
            'boltzmann': self._boltzmannAction,
            'ucb': self._ucbAction
        }[explorationStrategy]
        
        global printV, printD
        printV, printD = PrintIfVerbose(V), PrintIfDebug(D)

        self.eS = explorationStrategy
        self.eV = explorationValue
        self.aR = annealingRate
        self.alpha = alpha
        self.gamma = gamma
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        
        self.model = self.createModel()
        self.usingMR = False  # using memory replay
        self.usingTN = False  # using target network

        if memorySize is not None and batchSize is not None:
            printV(f'Agent will use replay memory of size {memorySize} and batch size {batchSize}.')
            self.memorySize = memorySize
            self.memory = deque(maxlen=self.memorySize)
            self.batchSize = batchSize
            self.usingMR = True
        elif memorySize is not None or batchSize is not None:
            raise ValueError('Both memorySize and batchSize must be specified if one of them is specified.')

        if targetUpdateFreq is not None:
            printV(f'Agent will use target network with update frequency {targetUpdateFreq}.')
            self.targetUpdateFreq = targetUpdateFreq
            self.targetModel = self.createModel()
            self.usingTN = True

        return

    def train(self, env: gym.Env, nEpisodes: int, V: bool = False) -> dict[str, np.ndarray]:
        """
        Trains the agent on the given environment for a given number of episodes.

        Params:
            - [Env] `env`: the environment to train on
            - [int] `nEpisodes`: number of episodes to train for
            - [bool] `V`: whether to print verbose output
        """

        R = np.zeros(nEpisodes, dtype=np.int16)
        A = np.zeros((nEpisodes, self.actionSpace), dtype=np.int16)

        for ep in prog(range(nEpisodes), V, f'training for {nEpisodes} episodes'):
            s, _ = env.reset()
            observations = ObservationSet()

            while True:
                a = self.getAction(s)
                s_, r, done, timedOut, _ = env.step(a)
                observation = Observation(s, a, r, s_, done)
                observations.add(observation)
                # skipped if not using memory replay
                self.remember(observation)
                s = s_

                if done or timedOut:
                    R[ep] = observations.totalReward
                    A[ep] = (observations.nLeft, observations.nRight)
                    self.learn(observations)
                    break

            # conditions are handled in the functions themselves
            self.anneal()
            self.replay()
            self.updateTargetModel(ep)

        env.close()
        self.resetModelWeights()

        return {'rewards': R, 'actions': A}

    ###########################
    # Model-related functions #
    ###########################

    def createModel(self) -> Sequential:
        """ Creates and compiles a basic neural network. """
        DenseHe = partial(Dense, kernel_initializer='he_uniform')
        
        model = Sequential()
        model.add(DenseHe(64, activation='relu', input_dim=self.stateSpace))
        model.add(DenseHe(64, activation='relu'))
        model.add(DenseHe(self.actionSpace, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def resetModelWeights(self) -> None:
        """ Resets the weights of the agent's model(s). """
        self.model.set_weights(self.createModel().get_weights())
        if self.usingTN:
            self.targetModel.set_weights(self.createModel().get_weights())
        return
    
    def learn(self, observations: ObservationSet) -> None:
        """ Updates the behaviour model's network weights for a given set of observations. """
        s, a, r, s_, done = observations.unpack()
        notDone = np.logical_not(done)
        target = r
        target[notDone] += self.gamma * np.amax(self.QTarget(s_), axis=1)[notDone]
        targetF = self.QBehaviour(s)
        targetF[np.arange(len(a)), a] = target
        self.model.fit(s, targetF, epochs=1, verbose=0)
        return

    #################
    # Memory replay #
    #################

    def remember(self, observation: Observation) -> None:
        """ Adds an observation to the replay memory. """
        if not self.usingMR:
            return
        if len(self.memory) == self.memorySize:
            self.memory.popleft()
        self.memory.append(observation)
        return
    
    def replay(self) -> None:
        """ Updates the behaviour model's network weights using a batch of observations from the replay memory. """
        if not self.usingMR:
            return
        if len(self.memory) < self.batchSize:
            return
        # TODO: use a prioritized replay buffer to prefer sampling from actions that are less explored
        batch = np.random.choice(self.memory, self.batchSize)
        self.learn(ObservationSet(batch))
        return

    ##################
    # Target network #
    ##################

    def updateTargetModel(self, ep: int) -> None:
        """ Updates the target model's network weights if it is time to do so. """
        if not self.usingTN or ep % self.targetUpdateFreq != 0:
            return
        # TODO: use a gradient clipping technique to prevent the target model from diverging too
        self.targetModel.set_weights(self.model.get_weights())
        return

    ###############
    # Q-functions #
    ###############

    @property
    def QBehaviour(self) -> callable:
        """ Returns the Q-function based on the behaviour policy. """
        return self._QBehaviour
    
    @property
    def QTarget(self) -> callable:
        """
        Returns the Q-function based on the target policy (if using target network).
        Otherwise it just returns the Q-function based on the behaviour policy.
        """
        return self._QTarget if self.usingTN else self._QBehaviour

    def _QBehaviour(self, state) -> np.ndarray:
        """ Returns the Q-values based on the behaviour policy for one or more states. """
        if len(state.shape) == 1:
            return self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return self.model.predict(state, verbose=0)
    
    def _QTarget(self, state) -> np.ndarray:
        """ Returns the Q-values based on the target policy for one or more states. """
        if len(state.shape) == 1:
            return self.targetModel.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return self.targetModel.predict(state, verbose=0)

    ##########################
    # Exploration strategies #
    ##########################

    def _epsilonGreedyAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the ε-greedy exploration strategy. """

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)
        Q = self.QBehaviour(state)
        printD(f'ε-greedy: Q = {Q}')
        return np.argmax(Q)
    
    def _boltzmannAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the Boltzmann exploration strategy. """

        Q = self.QBehaviour(state)
        Q /= self.tau; Q -= np.max(Q)
        printD(f'Boltzmann: Q = {Q}')
        return np.random.choice(self.actionSpace, p=np.exp(Q) / np.sum(np.exp(Q)))

    def _ucbAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the UCB exploration strategy. """

        Q = self.QBehaviour(state)
        Q += np.sqrt(self.zeta * np.log(np.sum(self.counts)) / (self.counts+0.001))
        printD(f'UCB: Q = {Q}')
        return np.argmax(Q)

    def anneal(self):
        """ We only anneal ε & τ, for UCB the exploration value (ζ) is fixed. """
        if self.zeta is None:
            self.eV = max(0.01, self.eV * self.aR)
        return
    
    @property
    def epsilon(self) -> float:
        """ Exploration value for e-greedy. """
        return self.eV if self.eS == 'e-greedy' else None
    
    @property
    def tau(self) -> float:
        """ Exploration value for Boltzmann. """
        return self.eV if self.eS == 'boltzmann' else None
    
    @property
    def zeta(self) -> float:
        """ Exploration value for UCB. """
        return self.eV if self.eS == 'ucb' else None

    ##############
    # Deprecated #
    ##############

    def learnSingle(self, observation: Observation) -> None:
        """ Updates the behaviour model's network weights for a given observation. """
        s, a, r, s_, done = observation.unpack()
        target = r
        if not done:
            target += self.gamma * np.amax(self.QTarget(s_)[0], axis=0)
        targetF = self.QBehaviour(s)
        targetF[0][a] = target
        self.model.fit(np.expand_dims(s, axis=0), targetF, epochs=1, verbose=0)
        return


def renderEpisodes(env: gym.Env, modelPath: str, numEpisodes: int = 10, V: bool = False) -> None:
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
