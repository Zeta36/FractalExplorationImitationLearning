import atexit
import multiprocessing
import sys
import traceback
from typing import Callable, List

import numpy

from fragile.distributed.ray import ray
from fragile.distributed.ray.env import Environment as RemoteEnvironment
from fragile.core.env import Environment
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import split_similar_chunks, StateDict
from fragile.optimize import Function


class RayEnv(Environment):
    def __init__(
        self, env_callable: Callable[[dict], Environment], n_workers: int, env_kwargs: dict = None,
    ):
        self.n_workers = n_workers
        self.envs: List[RemoteEnvironment] = [
            RemoteEnvironment.remote(env_callable=env_callable, env_kwargs=env_kwargs)
            for _ in range(n_workers)
        ]

    @property
    def states_shape(self) -> tuple:
        """Return the shape of the internal state of the :class:`Environment`."""
        shape = self.envs[0].get_data.remote("states_shape")
        return ray.get(shape)

    @property
    def observs_shape(self) -> tuple:
        """Return the shape of the observations state of the :class:`Environment`."""
        shape = self.envs[0].get_data.remote("observs_shape")
        return ray.get(shape)

    def get_params_dict(self) -> StateDict:
        params = self.envs[0].get_params_dict.remote()
        return ray.get(params)

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        split_env_states = [
            env.step.remote(model_states=ms, env_states=es)
            for env, ms, es in zip(
                self.envs,
                model_states.split_states(self.n_workers),
                env_states.split_states(self.n_workers),
            )
        ]
        env_states = ray.get(split_env_states)
        new_env_states: StatesEnv = StatesEnv.merge_states(env_states)
        return new_env_states

    def reset(
        self, batch_size: int = 1, env_states: StatesEnv = None, *args, **kwargs
    ) -> StatesEnv:
        reset = [
            env.reset.remote(batch_size=batch_size, env_states=env_states, *args, **kwargs)
            for env in self.envs
        ]
        return ray.get(reset)[0]


class _ExternalProcess:
    """
    Step environment in a separate process for lock free paralellism.
    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.

    Attributes:
        TARGET: Name of the function that will be applied.

    ..notes:
        This is mostly a copy paste from
        https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py,
        but it lets us set and read the environment state.

    """

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5
    TARGET = "step"

    def __init__(self, constructor):

        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._states_shape = None
        self._observs_shape = None

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        Args:
          name: Attribute to access.

        Returns:
          Value of the attribute.
        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        Args:
          name: Name of the method to call.
          *args: Positional arguments to forward to the method.
          **kwargs: Keyword arguments to forward to the method.

        Returns:
          Promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, blocking: bool = False, *args, **kwargs):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and n_repeat_actions as input.

        Args:
           blocking: If True, execute sequentially.
           args: Passed tot he target function.
           kwargs: passed to the target function.

        Returns:
            Return values of the target function.

        """
        promise = self.call(self.TARGET, *args, **kwargs)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking: bool = False, *args, **kwargs):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and n_repeat_actions as input.

        Args:
           blocking: If True, execute sequentially.
           args: Passed tot he target function.
           kwargs: passed to the target function.

        Returns:
            Return values of the target function.

        """
        promise = self.call("reset", *args, **kwargs)
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        Raises:
          Exception: An exception was raised inside the worker process.
          KeyError: The received message is of an unknown type.

        Returns:
          Payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, constructor, conn):
        """The process waits for actions and sends back environment results.
        Args:
          constructor: Constructor for the OpenAI Gym environment.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            env = constructor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class _BatchEnv:
    """Combine multiple environments to step them in batch.
    It is mostly a copy paste from
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
    that also allows to set and get the states.

    To step environments in parallel, environments must support a
        `blocking=False` argument to their step and reset functions that makes them
        return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """

    def __init__(self, envs, blocking):
        self._envs = envs
        self._blocking = blocking

    def __len__(self):
        """Number of combined environments."""
        return len(self._envs)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._envs[index]

    def __getattr__(self, name):
        """Forward unimplemented attributes to one of the original environments.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name one of the wrapped environments.
        """
        return getattr(self._envs[0], name)

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()

    def reset(self, batch_size: int = 1, env_states: StatesEnv = None, **kwargs) -> StatesEnv:
        results = [
            env.reset(self._blocking, batch_size=batch_size, env_states=env_states, **kwargs)
            for env in self._envs
        ]
        states = [result if self._blocking else result() for result in results]
        return states[0]


class _BatchEnvironment(_BatchEnv):
    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """Forward a batch of actions to the wrapped environments.
        Args:
          model_states: States representing the data to be used to act on the environment.
          env_states: States representing the data to be set in the environment.

        Returns:
          Batch of observations, rewards, and done flags.
        """
        split_states = self._make_transitions(model_states=model_states, env_states=env_states)
        states: StatesEnv = StatesEnv.merge_states(split_states)
        return states

    def _make_transitions(
        self, model_states: StatesModel, env_states: StatesEnv
    ) -> List[StatesEnv]:
        n_chunks = len(self._envs)
        results = [
            env.step(self._blocking, env_states=es, model_states=ms)
            for env, es, ms in zip(
                self._envs, env_states.split_states(n_chunks), model_states.split_states(n_chunks)
            )
        ]
        states = [result if self._blocking else result() for result in results]
        return states


class _ParallelEnvironment:
    """
    Wrap any environment to be stepped in parallel when step is called.

    """

    def __init__(self, env_callable, n_workers: int = 8, blocking: bool = False):
        self._env = env_callable()
        envs = [_ExternalProcess(constructor=env_callable) for _ in range(n_workers)]
        self._batch_env = _BatchEnvironment(envs, blocking)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`,
        but taking a list of states, actions and n_repeat_actions as input.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            :class:`StatesEnv` defining the new state of the :class:`Environment`.

        """
        return self._batch_env.step(env_states=env_states, model_states=model_states)

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the :class:`Environment` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`EnvStates` instance describing the state of the :class:`Environment`. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        return self._batch_env.reset(batch_size=batch_size, **kwargs)


class ParallelEnvironment(Environment):
    def __init__(
        self, env_callable: Callable[..., Environment], n_workers: int = 1, blocking: bool = False
    ):
        self.n_workers = n_workers
        self.blocking = blocking
        self.parallel_env = _ParallelEnvironment(
            env_callable=env_callable, n_workers=n_workers, blocking=blocking
        )
        self._local_env = env_callable()

    def __getattr__(self, item):
        return getattr(self._local_env, item)

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        return self.parallel_env.step(model_states=model_states, env_states=env_states)

    def reset(self, batch_size: int = 1, env_states: StatesEnv = None, **kwargs) -> StatesEnv:
        """
        Reset the environment and return an States class with batch_size copies \
        of the initial state.

        Args:
            batch_size: Number of walkers that the resulting state will have.
            env_states: States class used to set the environment to an arbitrary \
                        state.
            kwargs: Additional keyword arguments not related to environment data.

        Returns:
            States class containing the information of the environment after the \
             reset.

        """
        return self.parallel_env.reset(batch_size=batch_size, env_states=env_states, **kwargs)


class _ExternalFunction(_ExternalProcess):
    TARGET = "function"


class _BatchFunction(_BatchEnv):
    def step(self, points: numpy.ndarray):
        """Forward a batch of actions to the wrapped environments.
        Args:
          points: Batch of points that will be stepped.

        Raises:
          ValueError: Invalid actions.

        Returns:
          Batch of observations, rewards, and done flags.
        """
        rewards = self._make_transitions(points)
        try:
            rewards = numpy.stack(rewards)
        except BaseException as e:  # Lets be overconfident for once TODO: remove this.
            for obs in rewards:
                print(obs.shape)
        return numpy.concatenate([r.flatten() for r in rewards])

    def _make_transitions(self, points):
        chunks = len(self._envs)
        states_chunk = split_similar_chunks(points, n_chunks=chunks)
        results = [
            env.step(self._blocking, states_batch)
            for env, states_batch in zip(self._envs, states_chunk)
        ]
        rewards = [result if self._blocking else result() for result in results]
        return rewards


class _ParallelFunction:
    """
    Wrap any environment to be stepped in parallel when step_batch is called.

    """

    def __init__(self, env_callable, n_workers: int = 8, blocking: bool = False):
        self._env = env_callable()
        envs = [_ExternalFunction(constructor=env_callable) for _ in range(n_workers)]
        self._batch_env = _BatchFunction(envs, blocking)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`,
        but taking a list of states, actions and n_repeat_actions as input.

        Args:
            points: Batch of points that will be stepped.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)

        """
        return self._batch_env.step(points=points)

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the :class:`Function` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`EnvStates` instance describing the state of the :class:`Function`. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        return self._batch_env.reset(batch_size=batch_size, **kwargs)


class ParallelFunction(Function):
    def __init__(
        self, env_callable: Callable[..., Function], n_workers: int = 1, blocking: bool = False
    ):
        self.n_workers = n_workers
        self.blocking = blocking
        self.parallel_function = _ParallelFunction(
            env_callable=env_callable, n_workers=n_workers, blocking=blocking
        )
        self.local_function = env_callable()

    def __getattr__(self, item):
        return getattr(self.local_function, item)

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        new_points = model_states.actions + env_states.observs
        ends = self.calculate_end(points=new_points)
        rewards = self.parallel_function.step(new_points)

        updated_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards,
            ends=ends,
            batch_size=model_states.n,
        )
        return updated_states

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the :class:`Function` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`EnvStates` instance describing the state of the :class:`Function`. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        return self.parallel_function.reset(batch_size=batch_size, **kwargs)
