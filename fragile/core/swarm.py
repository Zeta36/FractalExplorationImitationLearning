import copy
from typing import Callable, List

import numpy

from fragile.core.base_classes import (
    BaseCritic,
    BaseEnvironment,
    BaseModel,
    BaseStateTree,
    BaseSwarm,
)
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import clear_output, Scalar
from fragile.core.walkers import StatesWalkers, Walkers


class Swarm(BaseSwarm):
    """
    The Swarm is in charge of performing a fractal evolution process.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to run the Swarm evolution algorithm.
    """

    def __init__(self, walkers: Callable = Walkers, *args, **kwargs):
        """Initialize a :class:`Swarm`."""
        self._use_tree = False
        self._prune_tree = False
        super(Swarm, self).__init__(walkers=walkers, *args, **kwargs)

    def __len__(self) -> int:
        return self.walkers.n

    def __repr__(self) -> str:
        return self.walkers.__repr__()

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
    def walkers(self) -> Walkers:
        """
        Access the :class:`Walkers` in charge of implementing the FAI \
        evolution process.
        """
        return self._walkers

    @property
    def best_found(self) -> numpy.ndarray:
        """Return the best state found in the current algorithm run."""
        return self.walkers.states.best_obs

    @property
    def best_reward_found(self) -> Scalar:
        """Return the best reward found in the current algorithm run."""
        return self.walkers.states.best_reward

    @property
    def critic(self) -> BaseCritic:
        """Return the :class:`Critic` of the walkers."""
        return self._walkers.critic

    def init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        tree: Callable = None,
        prune_tree: bool = True,
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
            dist_scale: Virtual reward exponent for the distance score.
            tree: class:`StatesTree` that keeps track of the visited states.
            prune_tree: If `tree` is `None` it has no effect. If true, \
                       store in the :class:`Tree` only the past history of alive \
                        walkers, and discard the branches with leaves that have \
                        no walkers.
            args: Passed to ``walkers_callable``.
            kwargs: Passed to ``walkers_callable``.

        Returns:
            None.

        """
        self._env: BaseEnvironment = env_callable()
        self._model: BaseModel = model_callable(self._env)

        model_params = self._model.get_params_dict()
        env_params = self._env.get_params_dict()
        self._walkers: Walkers = walkers_callable(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )
        self._use_tree = tree is not None
        self.tree: BaseStateTree = tree() if self._use_tree else None
        self._prune_tree = prune_tree
        self.epoch = 0

    def reset(
        self,
        walkers_states: StatesWalkers = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
    ):
        """
        Reset the :class:`fragile.Walkers`, the :class:`Environment`, the \
        :class:`Model` and clear the internal data to start a new search process.

        Args:
            model_states: :class:`StatesModel` that define the initial state of \
                          the :class:`Model`.
            env_states: :class:`StatesEnv` that define the initial state of \
                        the :class:`Environment`.
            walkers_states: :class:`StatesWalkers` that define the internal \
                            states of the :class:`Walkers`.
        """
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states
        model_states = (
            self.model.reset(batch_size=self.walkers.n, env_states=env_states)
            if model_states is None
            else model_states
        )
        model_states.update(init_actions=model_states.actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)
        if self._use_tree:
            root_ids = numpy.array([self.tree.ROOT_HASH] * self.walkers.n)
            self.walkers.states.id_walkers = root_ids
            self.tree.reset(
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=walkers_states,
            )
            self.update_tree(root_ids.tolist())

    def run(
        self,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        print_every: int = 1e100,
    ):
        """
        Run a new search process.

        Args:
            model_states: :class:`StatesModel` that define the initial state of \
                          the :class:`Model`.
            env_states: :class:`StatesEnv` that define the initial state of \
                        the :class:`Function`.
            walkers_states: :class:`StatesWalkers` that define the internal \
                            states of the :class:`Walkers`.
            print_every: Display the algorithm progress every ``print_every`` epochs.

        Returns:
            None.

        """
        self.reset(model_states=model_states, env_states=env_states, walkers_states=walkers_states)
        self.epoch = 0
        while not self.calculate_end_condition():
            try:
                self.run_step()
                if self.epoch % print_every == 0 and self.epoch > 0:
                    print(self)
                    clear_output(True)
                self.epoch += 1
            except KeyboardInterrupt:
                break

    def calculate_end_condition(self) -> bool:
        """Implement the logic for deciding if the algorithm has finished. \
        The algorithm will stop if it returns True."""
        return self.walkers.calculate_end_condition()

    def step_and_update_best(self) -> None:
        """
        Make the positions of the walkers evolve and keep track of the new states found.

        It also keeps track of the best state visited.
        """
        self.walkers.update_best()
        self.walkers.fix_best()
        self.step_walkers()

    def balance_and_prune(self) -> None:
        """
        Calculate the virtual reward and perform the cloning process.

        It also updates the :class:`Tree` data structure that takes care of \
        storing the visited states.
        """
        self.walkers.balance()
        new_ids = set(self.walkers.states.id_walkers.tolist())
        self.prune_tree(leaf_nodes=new_ids)

    def run_step(self) -> None:
        """
        Compute one iteration of the :class:`Swarm` evolution process and \
        update all the data structures.
        """
        self.step_and_update_best()
        self.balance_and_prune()
        self.walkers.fix_best()

    def step_walkers(self) -> None:
        """
        Make the walkers evolve to their next state sampling an action from the \
        :class:`Model` and applying it to the :class:`Environment`.
        """
        self.walkers.n_iters += 1
        model_states = self.walkers.model_states
        env_states = self.walkers.env_states

        states_ids = (
            copy.deepcopy(self.walkers.states.id_walkers).astype(int).flatten().tolist()
            if self._use_tree
            else None
        )

        model_states = self.model.predict(
            env_states=env_states, model_states=model_states, walkers_states=self.walkers.states
        )
        env_states = self.env.step(model_states=model_states, env_states=env_states)
        self.walkers.update_states(
            env_states=env_states, model_states=model_states, end_condition=env_states.ends
        )
        self.walkers.update_ids()
        self.update_tree(states_ids)

    def update_tree(self, states_ids: List[int]) -> None:
        """
        Add a list of walker states represented by `states_ids` to the :class:`Tree`.

        Args:
            states_ids: list containing the ids of the new states added.
        """
        if self._use_tree:
            self.tree.add_states(
                parent_ids=states_ids,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=self.walkers.states,
                n_iter=int(self.walkers.n_iters),
            )

    def prune_tree(self, leaf_nodes) -> None:
        """
        Remove all the branches that are do not have alive walkers at their leaf nodes.

        Args:
            leaf_nodes: ids of the new leaf nodes.

        """
        if self._prune_tree and self._use_tree:
            self.tree.prune_tree(alive_leafs=leaf_nodes, from_hash=True)


class NoBalance(Swarm):
    """Swarm that does not perform the cloning process."""

    def balance_and_prune(self):
        """Do noting."""
        pass

    def calculate_end_condition(self):
        """Finish after reaching the maximum number of epochs."""
        return self.epoch > self.walkers.max_iters
