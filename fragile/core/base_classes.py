from typing import Callable, List, Union

import numpy as np

from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import RANDOM_SEED, random_state, StateDict


class StatesOwner:
    """
    Every class that stores its data in :class:`States` must inherit \
    from this class.
    """

    random_state = random_state
    STATE_CLASS = States

    @classmethod
    def seed(cls, seed: int = RANDOM_SEED):
        """Set the random seed of the random number generator."""
        cls.random_state.seed(seed)

    @classmethod
    def get_params_dict(cls) -> StateDict:
        """
        Return an state_dict to be used for instantiating an States class.

        In order to define the tensors, a state_dict dictionary needs to be specified \
        using the following structure::

            import numpy as numpy
            state_dict = {"name_1": {"size": tuple([1]),
                                     "dtype": numpy.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that \
        will be accessed using the name_1 attribute of the class.
        """
        raise NotImplementedError

    def create_new_states(self, batch_size: int) -> "StatesOwner.STATE_CLASS":
        """Create new states of given batch_size to store the data of the class."""
        return self.STATE_CLASS(state_dict=self.get_params_dict(), batch_size=batch_size)

    def states_from_data(self, batch_size: int, **kwargs) -> States:
        """
        Initialize a :class:`States` with the data provided as kwargs.

        Args:
            batch_size: Number of elements in the first dimension of the \
                       :class:`State` attributes.
            **kwargs: Attributes that will be added to the returned :class:`States`.

        Returns:
            A new :class:`States` created with the class ``params_dict`` updated \
            with the attributes passed as keyword arguments.

        """
        state = self.create_new_states(batch_size=batch_size)
        state.update(**kwargs)
        return state


class BaseStateTree:
    """Data structure in charge of storing the history of visited states of an algorithm run."""

    ROOT_ID = 0
    ROOT_HASH = 0

    def add_states(
        self,
        parent_ids: List[int],
        env_states: States = None,
        model_states: States = None,
        walkers_states: States = None,
        n_iter: int = None,
    ) -> None:
        """
        Update the history of the tree adding the necessary data to recreate a \
        the trajectories sampled by the :class:`Swarm`.

        Args:
            parent_ids: List of states hashes representing the parent nodes of \
                        the current states.
            env_states: :class:`StatesEnv` containing the data that will be \
                        saved as new leaves in the tree.
            model_states: :class:`StatesModel` containing the data that will be \
                        saved as new leaves in the tree.
            walkers_states: :class:`StatesWalkers` containing the data that will be \
                        saved as new leaves in the tree.
            n_iter: Number of iteration of the algorithm when the data was sampled.

        Returns:
            None

        """
        pass

    def reset(
        self,
        env_states: States = None,
        model_states: States = None,
        walkers_states: States = None,
    ) -> None:
        """
        Delete all the data currently stored and reset the internal state of \
        the tree.
        """
        pass

    def prune_tree(self, alive_leafs: set, from_hash: bool = False) -> None:
        """
        Remove the branches that do not have a walker in their leaves.

        Args:
            alive_leafs: Contains the ids  of the leaf nodes that are being \
                         expanded by the walkers.
            from_hash: from_hash: If ``True`` ``alive_leafs`` will be \
                      considered a set of hashes of states. If ``False`` it \
                      will be considered a set of node ids.

        Returns:
            None.

        """
        pass


class BaseCritic(StatesOwner):
    """
    Perform additional computation. It can be used in a :class:`Walkers` \
    or a :class:`Model`.
    """

    random_state = random_state

    @classmethod
    def get_params_dict(cls) -> StateDict:
        """
        Return an state_dict to be used for instantiating an States class.

        In order to define the tensors, a state_dict dictionary needs to be specified \
        using the following structure::

            import numpy as numpy
            state_dict = {"name_1": {"size": tuple([1]),
                                     "dtype": numpy.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that \
        will be accessed using the name_1 attribute of the class.
        """
        state_dict = {
            "critic_score": {"size": tuple([1]), "dtype": np.float32},
        }
        return state_dict

    def calculate(
        self,
        batch_size: int = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: :class:`StatesModel` corresponding to the :class:`Model` data.
            env_states: :class:`StatesEnv` corresponding to the :class:`Environment` data.
            walkers_states: :class:`StatesWalkers` corresponding to the :class:`Walkers` data.

        Returns:
            States containing the the internal state of the :class:`BaseCritic`

        """
        raise NotImplementedError

    def reset(
        self,
        batch_size: int = 1,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs
    ) -> Union[States, None]:
        """
        Restart the `Critic` and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            env_states: :class:`StatesEnv` corresponding to the :class:`Environment` data.
            walkers_states: :class:`StatesWalkers` corresponding to the :class:`Walkers` data.
            args: Additional arguments not related to :class:`BaseCritic` data.
            kwargs: Additional keyword arguments not related to :class:`BaseCritic` data.

        Returns:
            States containing the information of the current state of the \
            :class:`BaseCritic` (after the reset).

        """
        pass

    def update(
        self,
        batch_size: int = 1,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs
    ) -> Union[States, None]:
        """
        Update the :class:`BaseCritic` internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            env_states: :class:`StatesEnv` corresponding to the :class:`Environment` data.
            walkers_states: :class:`StatesWalkers` corresponding to the :class:`Walkers` data.
            args: Additional arguments not related to :class:`BaseCritic` data.
            kwargs: Additional keyword arguments not related to :class:`BaseCritic` data.

        Returns:
            States containing the information of the current state of the \
            :class:`BaseCritic`.

        """
        pass


class BaseEnvironment(StatesOwner):
    """
    The Environment is in charge of stepping the walkers, acting as an state \
    transition function.

    For every different problem a new Environment needs to be implemented \
    following the :class:`BaseEnvironment` interface.

    """

    STATE_CLASS = StatesEnv

    def get_params_dict(self) -> StateDict:
        """
        Return an state_dict to be used for instantiating the states containing \
        the data describing the Environment.

        In order to define the arrays, a state_dict dictionary needs to be specified \
        using the following structure::

            import numpy as numpy
            # Example of an state_dict for planning.
            state_dict = {
                "states": {"size": self._env.get_state().shape, "dtype": numpy.int64},
                "observs": {"size": self._env.observation_space.shape, "dtype": numpy.float32},
                "rewards": {"dtype": numpy.float32},
                "ends": {"dtype": numpy.bool_},
            }

        """
        raise NotImplementedError

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Step the environment for a batch of walkers.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            States representing the next state of the environment and all \
            the needed information.

        """
        raise NotImplementedError

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
        raise NotImplementedError


class BaseModel(StatesOwner):
    """
    The model is in charge of calculating how the walkers will act on the \
    Environment, effectively working as a policy.
    """

    STATE_CLASS = StatesModel

    def get_params_dict(self) -> StateDict:
        """
        Return an state_dict to be used for instantiating the states containing \
        the data describing the Model.

        In order to define the arrays, a state_dict dictionary needs to be \
        specified using the following structure::

            import numpy as numpy
            # Example of an state_dict for a DiscreteUniform Model.
            n_actions = 10
            state_dict = {"actions": {"size": (n_actions,),
                                      "dtype": numpy.float32,
                                   },
                          "critic": {"size": tuple([n_actions]),
                                 "dtype": numpy.float32,
                               },
                          }

        Where size is a tuple indicating the shape of the desired tensor, \
        that will be accessed using the actions attribute of the class.
        """
        raise NotImplementedError

    def reset(
        self, batch_size: int = 1, model_states: StatesModel = None, *args, **kwargs
    ) -> StatesModel:
        """
        Restart the model and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            args: Additional arguments not related to model data.
            kwargs: Additional keyword arguments not related to model data.

        Returns:
            States containing the information of the current state of the \
            model (after the reset).

        """
        raise NotImplementedError

    def predict(
        self,
        batch_size: int = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
    ) -> StatesModel:
        """
        Calculate States containing the data needed to interact with the environment.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Updated model_states with new model data.

        """
        raise NotImplementedError


class BaseWalkers(StatesOwner):
    """
    The Walkers is a data structure that takes care of all the data involved \
    in making a Swarm evolve.
    """

    random_state = random_state
    STATE_CLASS = StatesWalkers

    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        accumulate_rewards: bool = True,
    ):
        """
        Initialize a `BaseWalkers`.

        Args:
            n_walkers: Number of walkers. This is the number of states that will\
             be iterated in parallel.
            env_state_params: Dictionary to instantiate the :class:`StatesEnv`\
             of an :class:`Environment`.
            model_state_params: Dictionary to instantiate the :class:`StatesModel`\
             of a :class:`Model`.
            accumulate_rewards: If `True` accumulate the rewards after each step
                of the environment.

        """
        super(BaseWalkers, self).__init__()
        self.n_walkers = n_walkers
        self.id_walkers = None
        self.death_cond = None
        self._accumulate_rewards = accumulate_rewards
        self.env_states_params = env_state_params
        self.model_states_params = model_state_params

    def __len__(self) -> int:
        """Return length is the number of walkers."""
        return self.n

    @property
    def n(self) -> int:
        """Return the number of walkers."""
        return self.n_walkers

    @property
    def env_states(self) -> StatesEnv:
        """Return the States class where all the environment information is stored."""
        raise NotImplementedError

    @property
    def model_states(self) -> StatesModel:
        """Return the States class where all the model information is stored."""
        raise NotImplementedError

    @property
    def states(self) -> StatesWalkers:
        """Return the States class where all the model information is stored."""
        raise NotImplementedError

    def get_params_dict(self) -> StateDict:
        """Return the params_dict of the internal StateOwners."""
        state_dict = {
            name: getattr(self, name).get_params_dict()
            for name in {"states", "env_states", "model_states"}
        }
        return state_dict

    def update_states(
        self, env_states: StatesEnv = None, model_states: StatesModel = None, **kwargs
    ) -> None:
        """
        Update the States variables that do not contain internal data and \
        accumulate the rewards in the internal states if applicable.

        Args:
            env_states: States containing the data associated with the Environment.
            model_states: States containing data associated with the Environment.
            **kwargs: Internal states will be updated via keyword arguments.

        """
        raise NotImplementedError

    def reset(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
    ):
        """
        Reset a :class:`Walkers` and clear the internal data to start a \
        new search process.

        Restart all the variables needed to perform the fractal evolution process.

        Args:
            model_states: :class:`StatesModel` that define the initial state of the environment.
            env_states: :class:`StatesEnv` that define the initial state of the model.
            walkers_states: :class:`StatesWalkers` that define the internal states of the walkers.

        """
        raise NotImplementedError

    def balance(self):
        """Perform FAI iteration to clone the states."""
        raise NotImplementedError

    def calculate_distances(self):
        """Calculate the distances between the different observations of the walkers."""
        raise NotImplementedError

    def calculate_virtual_reward(self):
        """Apply the virtual reward formula to account for all the different goal scores."""
        raise NotImplementedError

    def calculate_end_condition(self) -> bool:
        """Return a boolean that controls the stopping of the iteration loop. \
        If True, the iteration process stops."""
        raise NotImplementedError

    def clone_walkers(self):
        """Sample the clone probability distribution and clone the walkers accordingly."""
        raise NotImplementedError

    def get_alive_compas(self) -> np.ndarray:
        """
        Return an array of indexes corresponding to an alive walker chosen \
        at random.
        """
        raise NotImplementedError


class BaseSwarm:
    """
    The Swarm implements the iteration logic to make the :class:`Walkers` evolve.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to create the algorithm execution loop.
    """

    def __init__(
        self,
        env: Callable,
        model: Callable,
        walkers: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`BaseSwarm`.

        Args:
            env: A callable that returns an instance of an Environment.
            model: A callable that returns an instance of a Model.
            walkers: A callable that returns an instance of BaseWalkers.
            n_walkers: Number of walkers of the swarm.
            reward_scale: Virtual reward exponent for the reward score.
            dist_scale:Virtual reward exponent for the distance score.
            *args: Additional args passed to init_swarm.
            **kwargs: Additional kwargs passed to init_swarm.

        """
        self._walkers = None
        self._model = None
        self._env = None
        self.tree = None
        self.epoch = 0

        self.init_swarm(
            env_callable=env,
            model_callable=model,
            walkers_callable=walkers,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

    @property
    def env(self) -> BaseEnvironment:
        """All the simulation code (problem specific) will be handled here."""
        return self._env

    @property
    def model(self) -> BaseModel:
        """
        All the policy and random perturbation code (problem specific) will \
        be handled here.
        """
        return self._model

    @property
    def walkers(self) -> BaseWalkers:
        """
        Access the :class:`Walkers` in charge of implementing the FAI \
        evolution process.
        """
        return self._walkers

    def reset(
        self,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
    ):
        """
        Reset a :class:`fragile.Swarm` and clear the internal data to start a \
        new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
        """
        raise NotImplementedError

    def run(
        self,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
    ):
        """
        Run a new search process until the stop condition is met.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.

        Returns:
            None.

        """
        raise NotImplementedError

    def step_walkers(self):
        """
        Make the walkers undergo a perturbation process in the swarm \
        :class:`Environment`.

        This function updates the :class:`StatesEnv` and the :class:`StatesModel`.
        """
        raise NotImplementedError

    def init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        """
        Initialize and set up all the necessary internal variables to run the swarm.

        This process involves instantiating the Swarm, the Environment and the \
        model.

        Args:
            env_callable: A callable that returns an instance of an
                :class:`fragile.Environment`.
            model_callable: A callable that returns an instance of a
                :class:`fragile.Model`.
            walkers_callable: A callable that returns an instance of
                :class:`fragile.Walkers`.
            n_walkers: Number of walkers of the swarm.
            reward_scale: Virtual reward exponent for the reward score.
            dist_scale:Virtual reward exponent for the distance score.
            args: Additional arguments passed to reset.
            kwargs: Additional keyword arguments passed to reset.

        Returns:
            None.

        """
        raise NotImplementedError


class BaseWrapper:
    """Generic wrapper to wrap any of the other classes."""

    def __init__(self, data, name: str = "_unwrapped"):
        """
        Initialize a :class:`BaseWrapper`.

        Args:
            data: Object that will be wrapped.
            name: Assign a custom attribute name to the wrapped object.

        """
        setattr(self, name, data)
        self.__name = name

    @property
    def unwrapped(self):
        """Access the wrapped object."""
        return getattr(self, self.__name)

    def __repr__(self):
        return self.unwrapped.__repr__()

    def __call__(self, *args, **kwargs):
        """Call the wrapped class."""
        return self.unwrapped.__call__(*args, **kwargs)

    def __str__(self):
        return self.unwrapped.__str__()

    def __len__(self):
        return self.unwrapped.__len__()

    def __getattr__(self, attr):
        orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
