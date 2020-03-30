from typing import Optional

import numpy as np

from fragile.core.base_classes import BaseCritic
from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import float_type, StateDict


class GaussianDt(BaseCritic):
    """
    Sample an additional vector of clipped gaussian random variables, and \
    stores it in an attribute called `dt`.
    """

    @classmethod
    def get_params_dict(cls) -> StateDict:
        """Return the dictionary with the parameters to create a new `GaussianDt` critic."""
        base_params = super(GaussianDt, cls).get_params_dict()
        params = {"dt": {"dtype": float_type}}
        base_params.update(params)
        return params

    def __init__(
        self, min_dt: float = 1.0, max_dt: float = 1.0, loc_dt: float = 0.01, scale_dt: float = 1.0
    ):
        """
        Initialize a :class:`GaussianDt`.

        Args:
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.

        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = loc_dt
        self.std_dt = scale_dt

    def calculate(
        self,
        batch_size: Optional[int] = None,
        model_states: Optional[StatesModel] = None,
        env_states: Optional[StatesEnv] = None,
        walkers_states: Optional[StatesWalkers] = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled time step.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        dt = self.random_state.normal(loc=self.mean_dt, scale=self.std_dt, size=batch_size)
        dt = np.clip(dt, self.min_dt, self.max_dt)
        states = self.states_from_data(batch_size=batch_size, critic_score=dt, dt=dt)
        return states
