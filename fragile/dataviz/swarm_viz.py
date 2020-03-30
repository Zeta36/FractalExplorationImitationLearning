"""Wrappers to visualize the internal data of the :class:`Swarm`."""
from fragile.core.base_classes import BaseWrapper
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.swarm import Swarm
from fragile.core.utils import clear_output
from fragile.dataviz.swarm_stats import (
    AtariBestFrame,
    BestReward,
    DistanceHistogram,
    DistanceLandscape,
    GridLandscape,
    KDELandscape,
    RewardHistogram,
    RewardLandscape,
    SummaryTable,
    SwarmLandscape,
    VirtualRewardHistogram,
    VirtualRewardLandscape,
    WalkersDensity,
)

ALL_SWARM_TYPES = (
    GridLandscape,
    DistanceLandscape,
    RewardLandscape,
    VirtualRewardLandscape,
    WalkersDensity,
    DistanceHistogram,
    VirtualRewardHistogram,
    RewardHistogram,
    BestReward,
    SummaryTable,
    AtariBestFrame,
)

ALL_PLOT_NAMES = tuple([plot.name for plot in ALL_SWARM_TYPES])

ALL_SWARM_PLOTS = dict(zip(ALL_PLOT_NAMES, ALL_SWARM_TYPES))


class SwarmViz(BaseWrapper):
    """Wrap a :class:`Swarm` to incorporate visualizations."""

    SWARM_STATS_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
        SummaryTable,
    )
    DEFAULT_COLUMNS = 3
    DEFAULT_PLOTS = "all"
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))

    def __init__(
        self,
        swarm: Swarm,
        display_plots="default",
        stream_interval: int = 100,
        use_embeddings: bool = True,
        margin_high=1.0,
        margin_low=1.0,
        n_points: int = 50,
        columns: int = DEFAULT_COLUMNS,
    ):
        """
        Initialize a :class:`SwarmViz`.

        Args:
            swarm: :class:`Swarm` that will be wrapped.
            display_plots: List of plots that will be displayed. It contains \
                           the ``name`` class attribute of the plots that will \
                           be displayed. By default it plots the plots defined \
                           in ``cls.SWARM_STATS_TYPES``.

            stream_interval: Stream data to the plots every ``stream_interval`` iterations.
            use_embeddings: Use the embeddings provided by the :class:`Walkers` \
                            critic if available.
            margin_high: Ignored. (Update pending)
            margin_low: Ignored. (Update pending)
            n_points: Number of points for each coordinate of the meshgrid dimensions.
            columns: Number of columns of the generated grid of visualizations.

        """
        super(SwarmViz, self).__init__(data=swarm, name="swarm")
        display_plots = self.DEFAULT_PLOTS if display_plots == "default" else display_plots
        self.display_plots = self.PLOT_NAMES if display_plots == "all" else display_plots
        self.plots = self._init_plots(
            use_embeddings=use_embeddings,
            margin_low=margin_low,
            margin_high=margin_high,
            n_points=n_points,
        )
        self.stream_interval = stream_interval
        self.columns = columns

    def __getattr__(self, item):
        return getattr(self.swarm, item)

    def __repr__(self):
        return self.swarm.__repr__()

    def _init_plots(self, use_embeddings, margin_low, margin_high, n_points):
        plots = {}
        for name, plot in self.SWARM_PLOTS.items():
            if name not in self.display_plots:
                continue
            if issubclass(plot, SwarmLandscape):
                invert_cmap = self.swarm.walkers.minimize if name == "reward_landscape" else False
                plots[name] = self.SWARM_PLOTS[name](
                    margin_high=margin_high,
                    n_points=n_points,
                    margin_low=margin_low,
                    use_embeddings=use_embeddings,
                    invert_cmap=invert_cmap,
                )
            else:
                plots[name] = self.SWARM_PLOTS[name]()
        return plots

    def run(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
        print_every: int = 1e100,
    ):
        """
        Run a new search process.

        Args:
            env_states: :class:`StatesEnv` that define the initial state of the model.
            model_states: :class:`StatesEModel that define the initial state of the environment.
            walkers_states: :class:`StatesWalkers` that define the internal states of the walkers.
            print_every: Display the algorithm progress every `print_every` epochs.
        Returns:
            None.

        """
        self.swarm.reset(
            model_states=model_states, env_states=env_states, walkers_states=walkers_states
        )
        self.swarm.epoch = 0
        while not self.calculate_end_condition():
            try:
                self.run_step()
                if self.swarm.epoch % print_every == 0 and self.epoch > 0:
                    print(self)
                    clear_output(True)
                self.swarm.epoch += 1
            except KeyboardInterrupt:
                break

    def run_step(self):
        """
        Compute one iteration of the :class:`Swarm` evolution process and \
        update all the data structures, and stream the data to the created plots.
        """
        self.swarm.run_step()
        if self.walkers.n_iters % self.stream_interval == 0:
            self.stream_plots()

    def stream_plots(self):
        """Stream the :class:`Swarm` data to the plots."""
        for viz in self.plots.values():
            viz.stream_data(self)

    def plot(self, ignore: list = None):
        """Plot a DynamicMap that will contained the streaming plots of the selected data."""
        ignore = ignore if ignore is not None else []
        plots = [p.plot for k, p in self.plots.items() if k not in ignore]
        plot = plots[0]
        for p in plots[1:]:
            plot = plot + p
        return plot.cols(self.columns)


class Summary(SwarmViz):
    """
    :class:`Summary` that plots a table containing information of the epoch, \
    best reward found and percentage of deaths and clones.

    It also plots the evolution of the best reward found. It can work with any \
    kind of :class:`Environment`.
    """

    SWARM_STATS_TYPES = (
        SummaryTable,
        BestReward,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))
    DEFAULT_COLUMNS = 2


class AtariViz(SwarmViz):
    """
    :class:`Summary` that plots the RGB frame corresponding to the best state \
    found, in addition to the summary table and best reward plot.

    It also plots the evolution of the best reward found. It can work only with \
    an :class:`AtariEnv`.
    """

    SWARM_STATS_TYPES = (
        SummaryTable,
        AtariBestFrame,
        BestReward,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))
    DEFAULT_COLUMNS = 2
    DEFAULT_PLOTS = ("summary_table", "best_frame", "best_reward")


class SwarmViz1D(SwarmViz):
    """
    :class:`SwarmViz` that plots all the one-dimensional plots: Histograms and \
    reward evolution curve. It can work with any kind of :class:`Environment`.
    """

    SWARM_STATS_TYPES = (
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
        SummaryTable,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))
    DEFAULT_COLUMNS = 2


class LandscapeViz(SwarmViz):
    """
    :class:`SwarmViz` that plots all the one-dimensional plots: Histograms and \
    reward evolution curve. It can work with any kind of :class:`Environment`.
    """

    SWARM_STATS_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        BestReward,
        SummaryTable,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))


class GridViz(SwarmViz):
    """:class:`SwarmViz` that also plots a :class:`Critic` that contains a grid \
    to discretize the search space."""

    SWARM_STATS_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        GridLandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))


class KDEViz(SwarmViz):
    """:class:`SwarmViz` that also plots a :class:`Critic` that performs a KDE of the walkers."""

    SWARM_STATS_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        KDELandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    PLOT_NAMES = tuple([plot.name for plot in SWARM_STATS_TYPES])
    SWARM_PLOTS = dict(zip(PLOT_NAMES, SWARM_STATS_TYPES))
