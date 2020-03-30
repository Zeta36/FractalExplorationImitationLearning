import numpy as np

from fragile.core.functions import relativize
from fragile.core.walkers import Walkers

# import line_profiler


class AtariWalkers(Walkers):
    """
    This Walkers incorporate an additional stopping mechanism for the walkers \
    that allows to set a maximum score, and finish if the a given game has been \
    completely cleared.
    """

    def __init__(self, max_reward: int = None, *args, **kwargs):
        """
        Initialize a :class:`AtariWalkers`.

        Args:
            max_reward: If the accumulated reward of the :class:`AtariWalkers` \
                        reaches this values the algorithm will stop.
            *args: :class:`Walkers` parameters.
            **kwargs: :class:`Walkers` parameters.

        """
        super(AtariWalkers, self).__init__(*args, **kwargs)
        self.max_reward = max_reward

    def calculate_end_condition(self) -> bool:
        """
        Process data from the current state to decide if the iteration process \
        should stop. It not only keeps track of the maximum number of iterations \
        and the death condition, but also keeps track if the game has been played \
        until it finished.

        Returns:
            Boolean indicating if the iteration process should be finished. ``True`` \
            means it should be stopped, and ``False`` means it should continue.

        """
        end = super(AtariWalkers, self).calculate_end_condition()
        return self.env_states.game_ends.all() or end


class MontezumaWalkers(Walkers):
    """
    Walkers class used to calculate distances on Uber's Montezuma environment \
    used in their Go-explore repository.
    """

    # @profile
    def calculate_distances(self) -> None:
        """Calculate the corresponding distance function for each state with \
        respect to another state chosen at random.

        The internal state is update with the relativized distance values.

        The distance is performed on the RAM memory of the Atari emulator
        """
        compas_ix = np.random.permutation(np.arange(self.n))
        # This unpacks RAMs from Uber Go-explore custom Montezuma environment
        rams = self.env_states.states.reshape(self.n, -1)[:, :-12].astype(np.uint8)
        vec = rams - rams[compas_ix]
        dist_ram = self.distance_function(vec, axis=1).flatten()
        distances = relativize(dist_ram)
        self.update_states(distances=distances, compas_dist=compas_ix)
