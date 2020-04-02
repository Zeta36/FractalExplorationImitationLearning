from typing import Iterable

from fragile.core import Swarm
import numpy

from dqfd.kerasrl.memory import (
    PartitionedRingBuffer,
    SumSegmentTree,
    MinSegmentTree,
    PartitionedMemory as KrlPartitionedMemory,
)


class SwarmReplayMemory:
    def __init__(self, max_size: int, names: Iterable[str], mode: str = "best"):
        self.max_len = max_size
        self.names = names
        self.mode = mode
        for name in names:
            setattr(self, name, None)

    def __len__(self):
        if getattr(self, self.names[0]) is None:
            return 0
        return len(getattr(self, self.names[0]))

    def memorize(self, swarm: Swarm):
        # extract data from the swarm
        if self.mode == "best":
            data = next(swarm.tree.iterate_branch(swarm.best_id, batch_size=-1, names=self.names))
        else:
            data = next(swarm.tree.iterate_nodes_at_random(batch_size=-1, names=self.names))
        # Concatenate the data to the current memory
        for name, val in zip(self.names, data):
            if len(val.shape) == 1:  # Scalar vectors are transformed to columns
                val = val.reshape(-1, 1)
            processed = (
                val if getattr(self, name) is None else numpy.vstack([val, getattr(self, name)])
            )
            if len(processed) > self.max_len:
                processed = processed[: self.max_len]
            setattr(self, name, processed)
        print("Memory now contains %s samples" % len(self))


class DQFDMemory(SwarmReplayMemory):
    def __init__(self, max_size: int):
        names = ["observs", "actions", "rewards", "oobs"]
        super(DQFDMemory, self).__init__(max_size=max_size, mode="best", names=names)

    def iterate_data(self):
        if len(self) == 0:
            raise ValueError("Memory is empty. Call memorize before iterating data.")
        for i in range(len(self)):
            vals = [numpy.squeeze(getattr(self, name)[i]) for name in self.names]
            yield vals


class PartitionedMemory(KrlPartitionedMemory):
    def __init__(
        self,
        limit,
        swarm_memory: DQFDMemory,
        alpha=0.4,
        start_beta=1.0,
        end_beta=1.0,
        steps_annealed=1,
        **kwargs
    ):
        pre_load_data = [
            (obs, action, reward, end) for obs, action, reward, end in swarm_memory.iterate_data()
        ]
        print("LEN PRELOAD DATA", len(pre_load_data))
        super(PartitionedMemory, self).__init__(
            pre_load_data=pre_load_data,
            limit=limit,
            alpha=alpha,
            start_beta=start_beta,
            end_beta=end_beta,
            steps_annealed=steps_annealed,
            **kwargs
        )
