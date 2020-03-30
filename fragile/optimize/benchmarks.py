import math
from typing import Callable

from numba import jit
import numpy as np

from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.env import Bounds, Function

"""
This file includes several test functions for optimization described here:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, 1).flatten()


def rastrigin(x: np.ndarray) -> np.ndarray:
    dims = x.shape[1]
    A = 10
    result = A * dims + np.sum(x ** 2 - A * np.cos(2 * math.pi * x), 1)
    return result.flatten()


def eggholder(tensor: np.ndarray) -> np.ndarray:
    x, y = tensor[:, 0], tensor[:, 1]
    first_root = np.sqrt(np.abs(x / 2.0 + (y + 47)))
    second_root = np.sqrt(np.abs(x - (y + 47)))
    result = -1 * (y + 47) * np.sin(first_root) - x * np.sin(second_root)
    return result


def styblinski_tang(x) -> np.ndarray:
    return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, 1) / 2.0


@jit(nopython=True)
def _lennard_fast(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    for i in range(npart):
        for j in range(npart):
            if i > j:
                r2 = np.sum((state[j, :] - state[i, :]) ** 2)
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                epot = epot + r6i * (r6i - 1.0)
    epot = epot * 4
    return epot


def lennard_jones(x: np.ndarray) -> np.ndarray:
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        try:
            result[i] = _lennard_fast(x[i])
        except ZeroDivisionError:
            result[i] = np.inf
    return result


def _one_random_lennard(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    i, j = 0, 0
    while i == j:
        i, j = np.random.randint(0, npart, size=2).tolist()
    r2 = np.sum((state[j, :] - state[i, :]) ** 2)
    r2i = 1.0 / r2
    r6i = r2i * r2i * r2i
    epot = epot + r6i * (r6i - 1.0)
    epot = epot * 4
    return epot


def random_lennard(x: np.ndarray):
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        try:
            result[i] = _one_random_lennard(x[i])
        except ZeroDivisionError:
            result[i] = np.inf


class OptimBenchmark(Function):

    benchmark = None
    best_state = None

    def __init__(self, shape: tuple, function: Callable):
        bounds = self.get_bounds(shape=shape)
        super(OptimBenchmark, self).__init__(bounds=bounds, function=function)

    @staticmethod
    def get_bounds(shape):
        raise NotImplementedError


class Sphere(OptimBenchmark):
    benchmark = 0.0

    def __init__(self, shape: tuple):
        super(Sphere, self).__init__(shape=shape, function=sphere)

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1000, 1000) for _ in range(shape[0])]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return np.zeros(self.shape)


class Rastrigin(OptimBenchmark):
    benchmark = 0

    def __init__(self, shape: tuple):
        super(Rastrigin, self).__init__(shape=shape, function=rastrigin)

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.12, 5.12) for _ in range(shape[0])]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return np.zeros(self.shape)


class EggHolder(OptimBenchmark):
    benchmark = -959.64066271

    def __init__(self, shape=None):
        super(EggHolder, self).__init__(shape=(2,), function=eggholder)

    @staticmethod
    def get_bounds(shape=None):
        bounds = [(-512, 512), (-512, 512)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return np.array([512.0, 404.2319])


class StyblinskiTang(OptimBenchmark):
    def __init__(self, shape: tuple):
        super(StyblinskiTang, self).__init__(shape=shape, function=styblinski_tang)

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.0, 5.0) for _ in range(shape[0])]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return np.ones(self.shape) * -2.903534

    @property
    def benchmark(self):
        return -39.16617 * self.shape[0]


class LennardJones(OptimBenchmark):
    # http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
    minima = {
        "2": -1,
        "3": -3,
        "4": -6,
        "5": -9.103852,
        "6": -12.712062,
        "7": -16.505384,
        "8": -19.821489,
        "9": -24.113360,
        "10": -28.422532,
        "11": -32.765970,
        "12": -37.967600,
        "13": -44.326801,
        "14": -47.845157,
        "15": -52.322627,
        "20": -77.177043,
        "25": -102.372663,
        "30": -128.286571,
        "38": -173.928427,
        "50": -244.549926,
        "100": -557.039820,
        "104": -582.038429,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, *args, **kwargs):
        self.n_atoms = n_atoms
        shape = (3 * n_atoms,)
        self.benchmark = [np.zeros(self.n_atoms * 3), self.minima.get(str(int(n_atoms)), 0)]
        super(LennardJones, self).__init__(shape=shape, function=lennard_jones)

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1.5, 1.5) for _ in range(shape[0])]
        return Bounds.from_tuples(bounds)


class RandomLennard(LennardJones):
    def __init__(self, *args, **kwargs):
        super(RandomLennard, self).__init__(*args, **kwargs)
        self.random_lennard = random_lennard

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

        rewards = self.random_lennard(new_points).flatten()
        ends = self.calculate_end(points=new_points)

        last_states = self.states_from_data(model_states.n, new_points, new_points, rewards, ends)
        return last_states

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Resets the environment to the start of a new episode and returns an
        States instance describing the state of the Environment.
        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            States instance describing the state of the Environment. The first
            dimension of the data tensors (number of walkers) will be equal to
            batch_size.
        """
        ends = np.zeros(batch_size, dtype=np.bool_)
        new_points = self.sample_bounds(batch_size=batch_size)
        rewards = self.random_lennard(new_points).flatten()
        new_states = self.states_from_data(batch_size, new_points, new_points, rewards, ends)
        return new_states
