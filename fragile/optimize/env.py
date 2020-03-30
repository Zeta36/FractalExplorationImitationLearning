from typing import Callable, Tuple, Union

import numpy
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import minimize

from fragile.core.env import Environment
from fragile.core.models import Bounds
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import Scalar


class Function(Environment):
    """
    Environment that represents an arbitrary mathematical function bounded in a \
    given interval.
    """

    def __init__(
        self, function: Callable[[numpy.ndarray], numpy.ndarray], bounds: Bounds,
    ):
        """
        Initialize a :class:`Function`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      scalar. This function is applied to a batch of walker \
                      observations.
            bounds: :class:`Bounds` that defines the domain of the function.

        """
        if not isinstance(bounds, Bounds):
            raise TypeError("Bounds needs to be an instance of Bounds, found {}".format(bounds))
        self.function = function
        self.bounds = bounds
        self.shape = self.bounds.shape
        super(Function, self).__init__(observs_shape=self.shape, states_shape=self.shape)

    @classmethod
    def from_bounds_params(
        cls,
        function: Callable,
        shape: tuple = None,
        high: Union[int, float, numpy.ndarray] = numpy.inf,
        low: Union[int, float, numpy.ndarray] = -numpy.inf,
    ) -> "Function":
        """
        Initialize a function defining its shape and bounds without using a :class:`Bounds`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      scalar. This function is applied to a batch of walker \
                      observations.
            shape: Input shape of the solution vector without taking into account \
                    the batch dimension. For example, a two dimensional function \
                    applied to a batch of 5 walkers will have shape=(2,), even though
                    the observations will have shape (5, 2)
            high: Upper bound of the function domain. If it's an scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the upper bound for each dimension.
            low: Lower bound of the function domain. If it's an scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the lower bound for each dimension.

        Returns:
            :class:`Function` with its :class:`Bounds` created from the provided arguments.

        """
        if (
            not isinstance(high, numpy.ndarray)
            and not isinstance(low, numpy.ndarray) is None
            and shape is None
        ):
            raise TypeError("Need to specify shape or high or low must be a numpy array.")
        bounds = Bounds(high=high, low=low, shape=shape)
        return Function(function=function, bounds=bounds)

    def __repr__(self):
        text = "{} with function {}, obs shape {},".format(
            self.__class__.__name__, self.function.__name__, self.shape,
        )
        return text

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Set the :class:`Function` to the target states and sums the actions \
        provided by the :class:`StatesEnv`.

        Args:
            model_states: :class:`StatesModel` corresponding to the :class:`Model` data.
            env_states: :class:`StatesEnv` containing the data where the function \
             will be evaluated.

        Returns:
            :class:`StatesEnv` containing the information that describes the \
            new states sampled.

        """
        new_points = model_states.actions + env_states.observs
        ends = self.calculate_end(points=new_points)
        rewards = self.function(new_points).flatten()

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
        ends = numpy.zeros(batch_size, dtype=numpy.bool_)
        new_points = self.sample_bounds(batch_size=batch_size)
        rewards = self.function(new_points).flatten()
        new_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards,
            ends=ends,
            batch_size=batch_size,
        )
        return new_states

    def calculate_end(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        Determine if a given batch of vectors lie inside the function domain.

        Args:
            points: Array of batched vectors that will be checked to lie inside \
                    the :class:`Function` bounds.

        Returns:
            Array of booleans of length batch_size (points.shape[0]) that will \
            be ``True`` if a given point of the batch lies outside the bounds, \
            and ``False`` otherwise.

        """
        return numpy.logical_not(self.bounds.points_in_bounds(points)).flatten()

    def sample_bounds(self, batch_size: int) -> numpy.ndarray:
        """
        Return a matrix of points sampled uniformly from the :class:`Function` \
        domain.

        Args:
            batch_size: Number of points that will be sampled.

        Returns:
            Array containing ``batch_size`` points that lie inside the \
            :class:`Function` domain, stacked across the first dimension.

        """
        new_points = numpy.zeros(tuple([batch_size]) + self.shape, dtype=numpy.float32)
        for i in range(batch_size):
            new_points[i, :] = self.random_state.uniform(
                low=self.bounds.low, high=self.bounds.high, size=self.shape
            )
        return new_points


class Minimizer:
    """Apply ``scipy.optimize.minimize`` to a :class:`Function`."""

    def __init__(self, function: Function, bounds=None, *args, **kwargs):
        """
        Initialize a :class:`Minimizer`.

        Args:
            function: :class:`Function` that will be minimized.
            bounds: :class:`Bounds` defining the domain of the minimization \
                    process. If it is ``None`` the :class:`Function` :class:`Bounds` \
                    will be used.
            *args: Passed to ``scipy.optimize.minimize``.
            **kwargs: Passed to ``scipy.optimize.minimize``.

        """
        self.env = function
        self.function = function.function
        self.bounds = self.env.bounds if bounds is None else bounds
        self.args = args
        self.kwargs = kwargs

    def minimize(self, x: numpy.ndarray):
        """
        Apply ``scipy.optimize.minimize`` to a single point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Optimization result object returned by ``scipy.optimize.minimize``.

        """

        def _optimize(_x):
            try:
                _x = _x.reshape((1,) + _x.shape)
                y = self.function(_x)
            except (ZeroDivisionError, RuntimeError):
                y = numpy.inf
            return y

        bounds = ScipyBounds(
            ub=self.bounds.high if self.bounds is not None else None,
            lb=self.bounds.low if self.bounds is not None else None,
        )
        return minimize(_optimize, x, bounds=bounds, *self.args, **self.kwargs)

    def minimize_point(self, x: numpy.ndarray) -> Tuple[numpy.ndarray, Scalar]:
        """
        Minimize the target function passing one starting point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Tuple containing a numpy array representing the best solution found, \
            and the numerical value of the function at that point.

        """
        optim_result = self.minimize(x)
        point = optim_result["x"]
        reward = float(optim_result["fun"])
        return point, reward

    def minimize_batch(self, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Minimize a batch of points.

        Args:
            x: Array representing a batch of points to be optimized, stacked \
               across the first dimension.

        Returns:
            Tuple of arrays containing the local optimum found for each point, \
            and an array with the values assigned to each of the points found.

        """
        result = numpy.zeros_like(x)
        rewards = numpy.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            new_x, reward = self.minimize_point(x[i, :])
            result[i, :] = new_x
            rewards[i, :] = float(reward)
        return result, rewards


class MinimizerWrapper(Function):
    """
    Wrapper that applies a local minimization process to the observations \
    returned by a :class:`Function`.
    """

    def __init__(self, function: Function, *args, **kwargs):
        """
        Initialize a :class:`MinimizerWrapper`.

        Args:
            function: :class:`Function` to be minimized after each step.
            *args: Passed to the internal :class:`Optimizer`.
            **kwargs: Passed to the internal :class:`Optimizer`.

        """
        self.env = function
        self.minimizer = Minimizer(function=self.env, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def __repr__(self):
        return self.env.__repr__()

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Perform a local optimization process to the observations returned after \
        calling ``step`` on the wrapped :class:`Function`.

        Args:
            model_states: :class:`StatesModel` corresponding to the :class:`Model` data.
            env_states: :class:`StatesEnv` containing the data where the function \
             will be evaluated.

        Returns:
            States containing the information that describes the new state of \
            the :class:`Function`.

        """
        env_states = super(MinimizerWrapper, self).step(
            model_states=model_states, env_states=env_states
        )
        new_points, rewards = self.minimizer.minimize_batch(env_states.observs)
        ends = numpy.logical_not(self.bounds.points_in_bounds(new_points)).flatten()
        updated_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards.flatten(),
            ends=ends,
            batch_size=model_states.n,
        )
        return updated_states
