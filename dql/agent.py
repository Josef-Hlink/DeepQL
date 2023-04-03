""" Houses the DQLAgent class. """

from collections import deque
from functools import partial

from dql.utils.helpers import prog
from dql.utils.observations import Observation, ObservationSet

from psutil import Process
import numpy as np
import gym
from tensorflow import convert_to_tensor as toTensor
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class DQLAgent:
    """
    A Deep Q-Learning agent that is capable of memory replay and double DQN (using a target network).
    """

    def __init__(self,
        explorationStrategy: str, explorationValue: float,
        alpha: float, gamma: float, annealingRate: float, batchSize: int,
        memorySize: int = None, targetUpdateFreq: int = None,
        actionSpace: int = 2, stateSpace: int = 4
    ) -> None:
        """
        Initializes a Deep Q-Learning agent.
        
        Params:
            - [str]   `explorationStrategy`: one of 'e-greedy', 'boltzmann', 'ucb'
            - [float] `explorationValue`: (initial) value of the exploration parameter, e.g. epsilon for e-greedy
            - [float] `alpha`: learning rate for the neural network
            - [float] `gamma`: discount factor for future rewards
            - [float] `annealingRate`: rate at which the exploration parameter is annealed
            - [int]   `batchSize`: size of the batches used in both memory replay and regular training
            - [int]   `memorySize`: size of the replay memory
                * if specified, the agent will use memory replay
            - [int]   `targetUpdateFreq`: number of episodes after which the target network is updated
                * if specified, the agent will be a double DQN agent
            - [int]   `actionSpace`: number of possible actions in the environment
            - [int]   `stateSpace`: number of parameters that represent a state in the environment
        """

        self.getAction: function = {
            'e-greedy': self._epsilonGreedyAction,
            'boltzmann': self._boltzmannAction,
            'ucb': self._ucbAction
        }[explorationStrategy]

        self.eS = explorationStrategy
        self.eV = explorationValue
        self.eVbkp = explorationValue
        self.aR = annealingRate
        self.alpha = alpha
        self.gamma = gamma
        self.batchSize = batchSize
        self.observationBuffer = ObservationSet()
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        
        self.model = self.createModel()
        self.usingMR = False  # using memory replay
        self.usingTN = False  # using target network

        if memorySize is not None:
            self.memorySize = memorySize
            self.memory = deque(maxlen=self.memorySize)
            self.usingMR = True

        if targetUpdateFreq is not None:
            self.targetUpdateFreq = targetUpdateFreq
            self.targetModel = self.createModel()
            self.usingTN = True

        self.memoryLeakage = 0  # will be stored as number of bytes

        return

    def train(self, env: gym.Env, nEpisodes: int, V: bool = False) -> dict[str, np.ndarray]:
        """
        Trains the agent on the given environment for a given number of episodes.

        Params:
            - [Env] `env`: the environment to train on
            - [int] `nEpisodes`: number of episodes to train for
            - [bool] `V`: whether to print verbose output
        """

        self.R = np.zeros(nEpisodes, dtype=np.int16)
        self.A = np.zeros((nEpisodes, self.actionSpace), dtype=np.int16)

        for ep in prog(range(nEpisodes), V, f'training for {nEpisodes} episodes'):
            s, _ = env.reset()
            observations = ObservationSet()

            while True:
                a = self.getAction(s)
                s_, r, done, timedOut, _ = env.step(a)
                self.A[ep, a] += 1
                self.R[ep] += r
                observation = Observation(s, a, r, s_, done)
                observations.add(observation)
                # skipped if not using memory replay
                self.remember(observation)
                s = s_

                if done or timedOut:
                    self.R[ep] = observations.totalReward
                    self.learn(observations)
                    break

            # conditions are handled in the functions themselves
            self.anneal()
            self.replay()
            self.updateTargetModel(ep)

        return {'rewards': self.R, 'actions': self.A}

    def reset(self) -> None:
        """ Resets all relevant agent's member variables. """
        self.eV = self.eVbkp
        self.observationBuffer = ObservationSet()
        self.model = self.createModel()
        if self.usingMR:
            self.memory.clear()
        if self.usingTN:
            self.targetModel = self.createModel()
        self.memoryLeakage = 0
        return

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
    
    def learn(self, observations: ObservationSet) -> None:
        """ Wrapper around the actual learn function to abstract away batch storage. """
        self.observationBuffer += observations
        while len(self.observationBuffer) >= self.batchSize:
            self._learn(self.observationBuffer[:self.batchSize])
            del self.observationBuffer[:self.batchSize]
        return

    def _learn(self, observations: ObservationSet) -> None:
        """ Updates the behaviour model's network weights for a given set of observations. """
        s, a, r, s_, done = observations.unpack()
        notDone = np.logical_not(done)
        target = r
        target[notDone] += self.gamma * np.amax(self.QTarget(s_), axis=1)[notDone]
        targetF = self.QBehaviour(s)
        targetF[np.arange(len(a)), a] = target
        mB = Process().memory_info().rss
        self.model.fit(toTensor(s), toTensor(targetF), epochs=1, batch_size=len(observations), verbose=0)
        mA = Process().memory_info().rss
        self.memoryLeakage += mA - mB
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
        self._learn(ObservationSet(batch))
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
        mB = Process().memory_info().rss
        if len(state.shape) == 1:
            res = self.model.predict(toTensor(np.expand_dims(state, axis=0)), verbose=0)[0]
        else:
            res = self.model.predict(toTensor(state), verbose=0)
        mA = Process().memory_info().rss
        self.memoryLeakage += mA - mB
        return res
    
    def _QTarget(self, state) -> np.ndarray:
        """ Returns the Q-values based on the target policy for one or more states. """
        # if len(state.shape) == 1:
        #     return self.targetModel.predict(toTensor(np.expand_dims(state, axis=0)), verbose=0)[0]
        # return self.targetModel.predict(toTensor(state), verbose=0)
        mB = Process().memory_info().rss
        if len(state.shape) == 1:
            res = self.targetModel.predict(toTensor(np.expand_dims(state, axis=0)), verbose=0)[0]
        else:
            res = self.targetModel.predict(toTensor(state), verbose=0)
        mA = Process().memory_info().rss
        self.memoryLeakage += mA - mB
        return res

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
