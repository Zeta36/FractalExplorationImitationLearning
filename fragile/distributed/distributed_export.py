from typing import Callable


from fragile.distributed.export_swarm import BestWalker
from fragile.distributed.ray import ray
from fragile.distributed.ray.export_swarm import (
    ExportParamServer as RemoteParamServer,
    ExportSwarm as RemoteExportSwarm,
)


class DistributedExport:
    """
    Run a search process that exchanges :class:`ExportSwarm`.

    This search process uses a :class:`ParamServer` to exchange the specified \
    number o walkers between different :class:`ExportSwarm`.
    """

    def __init__(
        self,
        swarm: Callable,
        n_swarms: 2,
        n_import: int = 2,
        n_export: int = 2,
        export_best: bool = True,
        import_best: bool = True,
        max_len: int = 20,
        add_global_best: bool = True,
        swarm_kwargs: dict = None,
    ):
        """
        Initialize a :class:`DistributedExport`.

        Args:
            swarm: Callable that returns a :class:`Swarm`. Accepts keyword \
                   arguments defined in ``swarm_kwargs``.
            n_swarms: Number of :class:`ExportSwarm` that will be used in the \
                      to run the search process.
            n_import: Number of walkers that will be imported from an external \
                      :class:`ExportedWalkers`.
            n_export: Number of walkers that will be exported as :class:`ExportedWalkers`.
            export_best: The best walkers of the :class:`Swarm` will always be exported.
            import_best: The best walker of the imported :class:`ExportedWalkers` \
                         will be compared to the best walkers of the \
                         :class:`Swarm`. If it improves the current best value \
                         found, the best walker of the :class:`Swarm` will be updated.
            max_len: Maximum number of :class:`ExportedWalkers` that the \
                     :class:`ParamServer` will keep in its buffer.
            add_global_best: Add the best value found during the search to all \
                             the exported walkers that the :class:`ParamServer` \
                             returns.
            swarm_kwargs: Dictionary containing keyword that will be passed to ``swarm``.

        """
        self.swarms = [
            RemoteExportSwarm.remote(
                swarm=swarm,
                n_export=n_export,
                n_import=n_import,
                import_best=import_best,
                export_best=export_best,
                swarm_kwargs=swarm_kwargs,
            )
            for _ in range(n_swarms)
        ]
        self.n_swarms = n_swarms
        self.minimize = ray.get(self.swarms[0].get_data.remote("minimize"))
        self.max_iters = ray.get(self.swarms[0].get_data.remote("max_iters"))
        self.reward_limit = ray.get(self.swarms[0].get_data.remote("reward_limit"))
        self.param_server = RemoteParamServer.remote(
            max_len=max_len, minimize=self.minimize, add_global_best=add_global_best
        )
        self.epoch = 0

    def __getattr__(self, item):
        return ray.get(self.swarms[0].get_data.remote(item))

    def get_best(self) -> BestWalker:
        """Return the best walkers found during the algorithm run."""
        return ray.get(self.param_server.get_data.remote("best"))

    def reset(self):
        """Reset the internal data of the swarms and parameter server."""
        self.epoch = 0
        reset_param_server = self.param_server.reset.remote()
        reset_swarms = [swarm.reset.remote() for swarm in self.swarms]
        ray.get(reset_param_server)
        ray.get(reset_swarms)

    def run(self, print_every=1e10):
        """Run the distributed search algorithm asynchronously."""
        self.reset()
        current_import_walkers = self.swarms[0].get_empty_export_walkers.remote()
        steps = {}
        for swarm in self.swarms:
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

        for i in range(self.max_iters * self.n_swarms):
            self.epoch = i
            ready_export_walkers, _ = ray.wait(list(steps))
            ready_export_walker_id = ready_export_walkers[0]
            swarm = steps.pop(ready_export_walker_id)

            # Compute and apply gradients.
            current_import_walkers = self.param_server.exchange_walkers.remote(
                ready_export_walker_id
            )
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

            if self.epoch % print_every == 0 and self.epoch > 0:
                # Evaluate the current model after every 10 updates.
                best = self.get_best()
                print("iter {} best_reward_found: {:.3f}".format(i, best.rewards))
