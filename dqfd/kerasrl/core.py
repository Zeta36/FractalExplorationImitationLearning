# -*- coding: utf-8 -*-
from typing import Callable, List
import warnings
from copy import deepcopy

import numpy as np
from plangym.core import GymEnvironment
from tensorflow.python.keras.callbacks import History

from dqfd.kerasrl.callbacks import (
    Callback,
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer,
)


class Agent:
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """

    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0
        self.history_callback = History()
        self.callbacks: CallbackList = None
        self.compiled = False

    def get_config(self):
        """Configuration of the agent for serialization.

        # Returns
            Dictionnary with agent configuration
        """
        return {}

    def initialize_callbacks(
        self,
        callbacks: List[Callback],
        env: GymEnvironment,
        verbose: int = 1,
        log_interval: int = 1e10,
        visualize: bool = True,
        on_test: bool = False,
        **kwargs,
    ):
        callbacks = [] if not callbacks else callbacks[:]
        # TODO: Change this parameter nonsense. Check is append can be used instead of list concat
        if verbose >= 1 and on_test:
            callbacks += [TestLogger()]
        elif verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        self.history_callback = History()
        callbacks += [self.history_callback]
        callback_list: CallbackList = CallbackList(callbacks)
        # if hasattr(callbacks, "set_model"):
        #    callback_list.set_model(self)
        # else:
        callback_list.set_model(self)
        callback_list.set_env(env)
        # if hasattr(callbacks, "set_params"):
        callback_list.set_params(kwargs)
        # else:
        #    callback_list._set_params(kwargs)
        self.callbacks: CallbackList = callback_list

    def start_new_episode(
        self, episode: int, start_step_policy: Callable, n_max_start_steps: int, env
    ):
        self.callbacks.on_episode_begin(episode)
        episode_step = np.int16(0)
        episode_reward = np.float32(0)
        reward = None
        done = None
        info = None

        # Obtain the initial observation by resetting the environment.
        self.reset_states()
        observation = deepcopy(env.reset(return_state=False))
        if self.processor is not None:
            observation = self.processor.process_observation(observation)
        assert observation is not None

        # Perform random starts at beginning of episode and do not record them into the experience.
        # This slightly changes the start position between games.
        n_random_start_steps = (
            0 if n_max_start_steps == 0 else np.random.randint(n_max_start_steps)
        )
        for _ in range(n_random_start_steps):
            if start_step_policy is None:
                action = env.action_space.sample()
            else:
                action = start_step_policy(observation)
            if self.processor is not None:
                action = self.processor.process_action(action)
            self.callbacks.on_action_begin(action)
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            if self.processor is not None:
                (observation, reward, done, info,) = self.processor.process_step(
                    observation, reward, done, info
                )
            episode_reward += reward
            self.callbacks.on_action_end(action)
            if done:
                warnings.warn(
                    "Env ended before {} random steps could "
                    "be performed at the start. You should probably"
                    " lower the `n_max_start_steps`"
                    " parameter.".format(n_random_start_steps)
                )
                observation = deepcopy(env.reset(return_state=False))
                if self.processor is not None:
                    observation = self.processor.process_observation(observation)
                break
        return episode_step, episode_reward, observation, reward, done, info

    def step_environment(self, env, action, action_repetition):
        reward = np.float32(0)
        accumulated_info = {}
        done = False
        for _ in range(action_repetition):
            self.callbacks.on_action_begin(action)
            observation, r, done, info = env.step(action)
            observation = deepcopy(observation)
            if self.processor is not None:
                observation, r, done, info = self.processor.process_step(
                    observation, r, done, info
                )
            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value
            self.callbacks.on_action_end(action)
            reward += r
            if done:
                break
        return observation, reward, done, accumulated_info

    def fit(
        self,
        env,
        n_training_steps,
        action_repetition=1,
        callbacks=None,
        verbose: int = 1,
        visualize=False,
        n_max_start_steps: int = 0,
        start_step_policy=None,
        log_interval=10000,
        n_max_episode_steps=None,
    ):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for
                                   details.
            n_training_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `kerasrl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`),
                              2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            n_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            n_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                "Your tried to fit your agent but it hasn't been compiled yet. "
                "Please call `compile()` before `fit()`."
            )
        if action_repetition < 1:
            raise ValueError("action_repetition must be >= 1, is {}".format(action_repetition))

        self.training = True
        # Initialize callbacks
        self.initialize_callbacks(
            callbacks=callbacks,
            env=env,
            verbose=verbose,
            log_interval=log_interval,
            visualize=visualize,
            n_training_steps=n_training_steps,
        )
        self._on_train_begin()
        episode = np.int16(0)
        self.step = np.int16(0)
        did_abort = False
        episode_step, episode_reward, observation, reward, done, info = self.start_new_episode(
            episode=episode,
            start_step_policy=start_step_policy,
            n_max_start_steps=n_max_start_steps,
            env=env,
        )

        try:
            while self.step < n_training_steps:

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                self.callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                observation, reward, done, accumulated_info = self.step_environment(
                    env=env, action=action, action_repetition=action_repetition
                )
                episode_reward += reward
                if n_max_episode_steps and episode_step >= n_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)

                step_logs = {
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "metrics": metrics,
                    "episode": episode,
                    "info": accumulated_info,
                }
                self.callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0.0, terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        "episode_reward": episode_reward,
                        "nb_episode_steps": episode_step,
                        "n_training_steps": self.step,
                    }
                    self.callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    (
                        episode_step,
                        episode_reward,
                        observation,
                        reward,
                        done,
                        info,
                    ) = self.start_new_episode(
                        episode=episode,
                        start_step_policy=start_step_policy,
                        n_max_start_steps=n_max_start_steps,
                        env=env,
                    )
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        self.callbacks.on_train_end(logs={"did_abort": did_abort})
        self._on_train_end()
        return self.history_callback

    def test(
        self,
        env,
        n_episodes=1,
        action_repetition=1,
        callbacks=None,
        visualize=True,
        n_max_episode_steps=None,
        n_max_start_steps=0,
        start_step_policy=None,
        verbose=1,
    ):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `kerasrl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            n_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            n_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                "Your tried to test your agent but it hasn't been "
                "compiled yet. Please call `compile()` before `test()`."
            )
        if action_repetition < 1:
            raise ValueError("action_repetition must be >= 1, is {}".format(action_repetition))

        self.training = False
        self.step = 0

        self.initialize_callbacks(
            callbacks=callbacks,
            env=env,
            verbose=verbose,
            visualize=visualize,
            n_episodes=n_episodes,
            on_test=True,
        )

        self._on_test_begin()
        self.callbacks.on_train_begin()
        for episode in range(n_episodes):
            episode_step, episode_reward, observation, reward, done, info = self.start_new_episode(
                episode=episode,
                start_step_policy=start_step_policy,
                n_max_start_steps=n_max_start_steps,
                env=env,
            )

            # Run the episode until we're done.
            while not done:
                self.callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                observation, reward, done, accumulated_info = self.step_environment(
                    env=env, action=action, action_repetition=action_repetition
                )
                if n_max_episode_steps and episode_step >= n_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "episode": episode,
                    "info": accumulated_info,
                }
                self.callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0.0, terminal=False)

            # Report end of episode.
            episode_logs = {
                "episode_reward": episode_reward,
                "n_training_steps": episode_step,
            }
            self.callbacks.on_episode_end(episode, episode_logs)
        self.callbacks.on_train_end()
        self._on_test_end()

        return self.history_callback

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.

        # Returns
            A list of the model's layers
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).

        # Returns
            A list of metric's names (string)
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        self.callbacks.on_train_begin()

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor:
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            observation (object): An observation as obtained by the environment

        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            info (dict): An info as obtained by the environment

        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

        # Arguments
            action (int): Action given to the environment

        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.

        # Arguments
            batch (list): List of states

        # Returns
            Processed list of states
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py


class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.

    To implement your own environment, you need to define the following methods:

    - `step`
    - `reset`
    - `render`
    - `close`

    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """

    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode="human", close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)


class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.

    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()
