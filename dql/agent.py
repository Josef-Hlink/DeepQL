""" Houses the DQLAgent class. """

from functools import partial

from dql.utils.helpers import prog
from dql.utils.observations import Observation, ObservationSet, ObservationQueue

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import History


class DQLAgent:
    """
    A Deep Q-Learning agent that is capable of memory replay and double DQN (using a target network).
    """

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingRate: float, batchSize: int,
        replayBufferSize: int = None, targetUpdateFreq: int = None,
        actionSpace: int = 2, stateSpace: int = 4
    ) -> None:
        """
        Initializes a Deep Q-Learning agent.
        
        Params:
            - [str]   `explorationStrategy`: one of 'egreedy', 'boltzmann', 'ucb'
            - [float] `explorationValue`: (initial) value of the exploration parameter, e.g. epsilon for e-greedy
            - [float] `alpha`: learning rate for the neural network
            - [float] `gamma`: discount factor for future rewards
            - [float] `annealingRate`: rate at which the exploration parameter is annealed
            - [int]   `batchSize`: size of the batches used in both memory replay and regular training
            - [int]   `replayBufferSize`: size of the experience replay buffer
                * if specified, the agent will use experience replay (`usingER` will be set to True)
            - [int]   `targetUpdateFreq`: number of episodes after which the target network is updated
                * if specified, the agent will be a double DQN agent (`usingTN` will be set to True)
            - [int]   `actionSpace`: number of possible actions in the environment
            - [int]   `stateSpace`: number of parameters that represent a state in the environment
        """

        self.getAction: function = {
            'egreedy': self._epsilonGreedyAction,
            'boltzmann': self._boltzmannAction,
            'ucb': self._ucbAction
        }[explorationStrategy]

        self.eS = explorationStrategy
        self.eV = explorationValue
        self.eV0 = explorationValue
        self.aR = annealingRate
        self.alpha = alpha
        self.gamma = gamma
        self.batchSize = batchSize
        self.buffer = ObservationSet()
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        
        self.model = self.createModel()
        self.usingER = False  # using experience replay
        self.usingTN = False  # using target network

        if replayBufferSize is not None:
            self.replayBufferSize = replayBufferSize
            self.replayBuffer = ObservationQueue(maxSize=replayBufferSize)
            self.usingER = True

        if targetUpdateFreq is not None:
            self.targetUpdateFreq = targetUpdateFreq
            self.targetModel = self.createModel()
            self.usingTN = True

        return

    ###############################
    # API for outside interaction #
    ###############################

    def train(self, env: gym.Env, nEpisodes: int, V: bool = False) -> dict[str, np.ndarray | list]:
        """
        Trains the agent on the given environment for a given number of episodes.

        Params:
            - [Env] `env`: the environment to train on
            - [int] `nEpisodes`: number of episodes to train for
            - [bool] `V`: whether to print verbose output

        Returns:
            - [dict] `results`: a dictionary containing the rewards, actions, and losses as arrays
                * [array] `rewards`: total reward per episode [shape: nEpisodes]
                * [array] `actions`: total number of times each action was taken per episode [shape: (nEpisodes, actionSpace)]
                * [array] `losses`: loss per update [shape: variable!]

        ! the length of the losses array is variable as it depends on the number of total observations made by the agent
        """

        # initialize arrays to store rewards, actions, and losses
        self.R = np.zeros(nEpisodes, dtype=np.int16)
        self.A = np.zeros((nEpisodes, self.actionSpace), dtype=np.int16)
        self.L = []

        for ep in prog(range(nEpisodes), V, f'training for {nEpisodes} episodes'):

            s, _ = env.reset()
            while True:
                # get action from policy
                a = self.getAction(s)
                # take action and observe reward and next state
                s_, r, done, truncated, _ = env.step(a)
                # housekeeping
                self.A[ep, a] += 1
                self.R[ep] += r
                o = Observation(s, a, r, s_, done)
                # update state
                s = s_
                
                # add observation to buffer(s)
                self.buffer.add(o)
                if self.usingER:
                    self.replayBuffer.add(o)

                if done or truncated:
                    break
            
            # learn from observations in buffer
            while len(self.buffer) >= self.batchSize:
                loss = self.updateModel(self.buffer[:self.batchSize])
                self.L.append(loss)
                del self.buffer[:self.batchSize]
            # experience replay
            if self.usingER:
                self.updateModel(self.replayBuffer.sample(size=self.batchSize))
            # target network update
            if self.usingTN and ep % self.targetUpdateFreq == 0:
                self.updateTargetModel()

            # anneal the exploration parameter
            self.anneal()

        return {'rewards': self.R, 'actions': self.A, 'losses': np.array(self.L, dtype=np.float32)}

    def reset(self) -> None:
        """ Resets all relevant agent's member variables. """
        # reset exploration parameter
        self.eV = self.eV0
        # reset memory
        self.buffer = ObservationSet()
        if self.usingER:
            self.replayBuffer = ObservationSet(maxSize=self.replayBufferSize)
        # reset model(s)
        self.model = self.createModel()
        if self.usingTN:
            self.targetModel = self.createModel()
        return
    
    #########
    # Model #
    #########

    def createModel(self) -> Sequential:
        """ Creates and compiles a basic neural network. """
        DenseHe = partial(Dense, kernel_initializer='he_uniform')
        
        model = Sequential()
        model.add(DenseHe(24, activation='relu', input_dim=self.stateSpace))
        model.add(DenseHe(24, activation='relu'))
        model.add(DenseHe(self.actionSpace, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def updateModel(self, observations: ObservationSet) -> float:
        """ Updates the behaviour model's network weights for a given set of observations. """
        # unpack all observations into their respective components (as numpy arrays)
        s, a, r, s_, done = observations.unpack()
        # this will be used for masking the target values
        notDone = np.logical_not(done)
        # initialize the target values to just the rewards
        G = r
        # mask out the target values for the terminal states, they will stay at just their rewards
        G[notDone] += self.gamma * np.amax(self.QTarget(s_, training=True), axis=1)[notDone]
        # simulate the current behaviour model
        Q = self.QBehaviour(s, training=True)
        # update the Q-values for the actions taken in the observations to the target values
        Q[np.arange(len(a)), a] = G
        # train the behaviour model
        history: History = self.model.fit(s, Q, epochs=1, batch_size=self.batchSize, verbose=0)
        return history.history['loss'][0]
    
    def updateTargetModel(self) -> None:
        """ Updates the target model's network weights. """
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
    
    def _QBehaviour(self, state: np.ndarray, training=False) -> np.ndarray:
        """ Returns the Q-values based on the behaviour policy for one or more states. """
        if len(state.shape) == 1:
            return self.model(np.expand_dims(state, axis=0), training=training).numpy()[0]
        return self.model(state, training=training).numpy()
    
    def _QTarget(self, state: np.ndarray, training=False) -> np.ndarray:
        """ Returns the Q-values based on the target policy for one or more states. """
        if len(state.shape) == 1:
            return self.targetModel(np.expand_dims(state, axis=0), training=training).numpy()[0]
        return self.targetModel(state, training=training).numpy()

    ##########################
    # Exploration strategies #
    ##########################

    def _epsilonGreedyAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the ε-greedy exploration strategy. """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)
        Q = self.QBehaviour(state)
        return np.argmax(Q)
    
    def _boltzmannAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the Boltzmann exploration strategy. """
        Q = self.QBehaviour(state)
        Q /= self.tau; Q -= np.max(Q)
        return np.random.choice(self.actionSpace, p=np.exp(Q) / np.sum(np.exp(Q)))

    def _ucbAction(self, state: np.ndarray) -> int:
        """ Returns an action based on the UCB exploration strategy. """
        actionCounts = np.sum(self.A, axis=0)
        if np.any(actionCounts == 0):
            return np.argwhere(actionCounts == 0)[0][0]
        Q = self.QBehaviour(state)
        Q += np.sqrt(self.zeta * np.log(np.sum(self.A)) / np.sum(self.A, axis=0))
        return np.argmax(Q)

    def anneal(self):
        """ We only anneal for ε or τ; for UCB the exploration value (ζ) is fixed. """
        if self.zeta is None:
            self.eV = max(0.01, self.eV * self.aR)
        return
    
    @property
    def epsilon(self) -> float:
        """ Exploration value for ε-greedy. """
        return self.eV if self.eS == 'egreedy' else None
    
    @property
    def tau(self) -> float:
        """ Exploration value for Boltzmann. """
        return self.eV if self.eS == 'boltzmann' else None
    
    @property
    def zeta(self) -> float:
        """ Exploration value for UCB. """
        return self.eV if self.eS == 'ucb' else None

