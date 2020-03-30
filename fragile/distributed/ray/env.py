from typing import Callable

import numpy

from fragile.core.env import Environment
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import StateDict
from fragile.distributed.ray import ray
from fragile.optimize import Function as SequentialFunction


@ray.remote
class Environment(Environment):
    """
    :class:`fragile.Environment` remote interface to be used with ray.

    Wraps a :class:`fragile.Environment` passed as a callable.
    """

    def __init__(self, env_callable: Callable[[dict], Environment], env_kwargs: dict = None):
        """
        Initialize a :class:`Environment`.

        Args:
            env_callable: Callable that returns a :class:`fragile.Environment`.
            env_kwargs: Passed to ``env_callable``.

        """
        env_kwargs = {} if env_kwargs is None else env_kwargs
        self.env = env_callable(**env_kwargs)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def get_data(self, name: str):
        """
        Get an attribute from the wrapped environment.

        Args:
            name: Name of the target attribute.

        Returns:
            Attribute from the wrapped :class:`fragile.Environment`.

        """
        return getattr(self.env, name)

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Step the wrapped :class:`fragile.Environment`.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            States representing the next state of the environment and all \
            the needed information.

        """
        step = self.env.step(model_states=model_states, env_states=env_states)
        return step

    def reset(
        self, batch_size: int = 1, env_states: StatesEnv = None, *args, **kwargs
    ) -> StatesEnv:
        """
        Reset the wrapped :class:`fragile.Environment` and return an States class \
        with batch_size copies of the initial state.

        Args:
           batch_size: Number of walkers that the resulting state will have.
           env_states: States class used to set the environment to an arbitrary \
                       state.
           args: Additional arguments not related to environment data.
           kwargs: Additional keyword arguments not related to environment data.

        Returns:
           States class containing the information of the environment after the \
            reset.

        """
        return self.env.reset(batch_size=batch_size, env_states=env_states, *args, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary of the wrapped :class:`fragile.Environment`."""
        return self.env.get_params_dict()


@ray.remote
class Function:
    """
    :class:`fragile.Function` remote interface to be used in with ray.

    Wraps a :class:`fragile.Function` passed as a callable.
    """

    def __init__(self, env_callable: Callable[..., SequentialFunction], env_kwargs: dict = None):
        """
        Initialize a :class:`Function`.

        Args:
            env_callable: env_callable: Callable that returns a :class:`fragile.Function`.
            env_kwargs: Passed to ``env_callable``.

        """
        self.function = (
            env_callable().function if env_kwargs is None else env_callable(**env_kwargs).function
        )

    def function(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        Call the wrapped :class:`Function`.``function``.

        Args:
            points: Array of batched points that will be passed tot he wrapped function.

        Returns:
            Array with the rewards assigned to each point of the batch.

        """
        return self.function(points)
