from fragile.core import DiscreteEnv, DiscreteUniform
from fragile.core.tree import HistoryTree
from fragile.core.swarm import Swarm
from fragile.distributed import ParallelEnv
from plangym import AtariEnvironment

from dqfd.fragile_learning.env import AtariEnvironment
from dqfd.fragile_learning.memory import DQFDMemory


class FragileRunner:
    def __init__(
        self,
        game_name: str,
        n_walkers: int = 32,
        max_epochs: int = 200,
        reward_scale: float = 2.0,
        distance_scale: float = 1.0,
        n_workers: int = 8,
        memory_size: int = 200,
        score_limit: float = 600,
    ):

        self.env = ParallelEnv(lambda: AtariEnvironment(game_name), n_workers=n_workers)
        self.n_actions = self.env.n_actions
        self.game_name = game_name
        self.env_callable = lambda: self.env
        self.model_callable = lambda env: DiscreteUniform(env=self.env)
        self.prune_tree = True
        # A bigger number will increase the quality of the trajectories sampled.
        self.n_walkers = n_walkers
        self.max_epochs = max_epochs  # Increase to sample longer games.
        self.reward_scale = reward_scale  # Rewards are more important than diversity.
        self.distance_scale = distance_scale
        self.minimize = False  # We want to get the maximum score possible.
        store_data = ["observs", "actions", "rewards", "oobs"]
        self.swarm = Swarm(
            model=self.model_callable,
            env=self.env_callable,
            tree=lambda: HistoryTree(names=store_data, prune=True),
            n_walkers=self.n_walkers,
            max_epochs=self.max_epochs,
            prune_tree=self.prune_tree,
            reward_scale=self.reward_scale,
            distance_scale=self.distance_scale,
            minimize=self.minimize,
            score_limit=score_limit,
        )
        self.memory = DQFDMemory(max_size=memory_size)

    def run(self):
        while len(self.memory) < self.memory.max_len - 1:
            print("Creating fractal replay memory...")
            _ = self.swarm.run()
            print("Max. fractal cum_rewards:", self.swarm.best_reward)
            self.memory.memorize(swarm=self.swarm)
