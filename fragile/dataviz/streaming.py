"""Contains the necessary logic to create ``holoviews`` plots that accept streaming data."""
from typing import Any, Callable, Tuple, Union

import holoviews
from holoviews.streams import Buffer, Pipe
import numpy
from scipy.interpolate import griddata

from fragile.core.utils import Scalar


class Plot:
    """Base class that manages the creation of ``holoviews`` plots."""

    name = ""

    def __init__(self, plot: Callable, data: Any = None, *args, **kwargs):
        """
        Initialize a :class:`Plot`.

        Args:
            plot: Callable that returns an holoviews plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
            args: Passed to ``opts``.
            kwargs: Passed to ``opts``.
        """
        self.plot = None
        self.init_plot(plot, data, *args, **kwargs)

    def get_plot_data(self, data: Any):
        """Perform the necessary data wrangling for plotting the data."""
        raise NotImplementedError

    def init_plot(self, plot: Callable, data=None, *args, **kwargs):
        """
        Initialize the holoviews plot.

        Args:
            plot: Callable that returns an holoviews plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
            args: Passed to ``opts``.
            kwargs: Passed to ``opts``.

        """
        data = self.get_plot_data(data)
        self.plot = plot(data)
        self.opts(*args, **kwargs)

    def opts(self, *args, **kwargs):
        """Update the plot parameters. Same as ``holoviews`` ``opts``."""
        if self.plot is None:
            return
        self.plot = self.plot.opts(*args, **kwargs)


class StreamingPlot(Plot):
    """Represents a an holoviews plot updated with streamed data."""

    name = ""

    def __init__(self, plot: Callable, stream=Pipe, data=None, *args, **kwargs):
        """
        Initialize a :class:`StreamingPlot`.

        Args:
            plot: Callable that returns an holoviews plot.
            stream: Class used to stream data to the plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
            args: Passed to ``opts``.
            kwargs: Passed to ``opts``.
        """
        self.data_stream = None
        self.epoch = 0
        self.init_stream(stream, data)
        super(StreamingPlot, self).__init__(plot=plot, *args, **kwargs)

    def get_plot_data(self, data):
        """Perform the necessary data wrangling for plotting the data."""
        return data

    def stream_data(self, data) -> None:
        """Stream data to the plot and keep track of how many times the data has been streamed."""
        data = self.get_plot_data(data)
        self.data_stream.send(data)
        self.epoch += 1

    def init_plot(self, plot: Callable, data=None, *args, **kwargs) -> None:
        """
        Initialize the holoviews plot to accept streaming data.

        Args:
            plot: Callable that returns an holoviews plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
            args: Passed to ``opts``.
            kwargs: Passed to ``opts``.

        """
        self.plot = holoviews.DynamicMap(plot, streams=[self.data_stream])
        self.opts(*args, **kwargs)

    def init_stream(self, stream, data=None):
        """Initialize the data stream that will be used to stream data to the plot."""
        self.epoch = 0
        data = self.get_plot_data(data)
        self.data_stream = stream(data=data)


class Table(StreamingPlot):
    """``holoviews.Table`` with data streaming capabilities."""

    name = "table"

    def __init__(self, data=None, stream=Pipe, *args, **kwargs):
        """
        Initialize a :class:`Table`.

        Args:
            data: Data to initialize the stream.
            stream: :class:`holoviews.stream` type. Defaults to :class:`Pipe`.
            *args: Passed to :class:`StreamingPlot`.
            **kwargs: Passed to :class:`StreamingPlot`.

        """
        super(Table, self).__init__(
            stream=stream, plot=holoviews.Table, data=data, *args, **kwargs
        )

    def opts(self, *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        self.plot = self.plot.opts(holoviews.opts.Table(*args, **kwargs))


class RGB(StreamingPlot):
    """``holoviews.RGB`` with data streaming capabilities."""

    name = "rgb"

    def __init__(self, data=None, *args, **kwargs):
        """Initialize a :class:`RGB`."""
        super(RGB, self).__init__(stream=Pipe, plot=holoviews.RGB, data=data, *args, **kwargs)

    def opts(
        self, xaxis=None, yaxis=None, *args, **kwargs,
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        self.plot = self.plot.opts(holoviews.opts.RGB(xaxis=xaxis, yaxis=yaxis, *args, **kwargs))


class Curve(StreamingPlot):
    """
    Create a ``holoviews.Curve`` plot that plots steaming data.

    The streaming process is handled using a :class:`Buffer`.
    """

    name = "curve"

    def __init__(
        self, buffer_length: int = 10000, index: bool = False, data=None,
    ):
        """
        Initialize a :class:`Curve`.

        Args:
            buffer_length: Maximum number of data points that will be displayed in the plot.
            index: Passed to the :class:`Buffer` that streams data to the plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
        """

        def get_stream(data):
            return Buffer(data, length=buffer_length, index=index)

        super(Curve, self).__init__(stream=get_stream, plot=holoviews.Curve, data=data)

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Curve(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                *kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )


class Histogram(StreamingPlot):
    """
    Create a ``holoviews.Histogram`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "histogram"

    def __init__(self, n_bins: int = 20, data=None):
        """
        Initialize a :class:`Histogram`.

        Args:
            n_bins: Number of bins of the histogram that will be plotted.
            data: Used to initialize the plot.
        """
        self.n_bins = n_bins
        self.xlim = (None, None)
        super(Histogram, self).__init__(stream=Pipe, plot=self.plot_histogram, data=data)

    @staticmethod
    def plot_histogram(data):
        """
        Plot the histogram.

        Args:
            data: Tuple containing (values, bins), xlim. xlim is a tuple \
                  containing two scalars that represent the limits of the x \
                  axis of the histogram.

        Returns:
            Histogram plot.

        """ ""
        plot_data, xlim = data
        return holoviews.Histogram(plot_data).redim(x=holoviews.Dimension("x", range=xlim))

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "count",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Histogram(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                *kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )

    def get_plot_data(
        self, data: numpy.ndarray
    ) -> Tuple[
        Tuple[numpy.ndarray, numpy.ndarray], Tuple[Union[Scalar, None], Union[Scalar, None]]
    ]:
        """
        Calculate the histogram of the streamed data.

        Args:
            data: Values used to calculate the histogram.

        Returns:
            Tuple containing (values, bins), xlim. xlim is a tuple \
                  containing two scalars that represent the limits of the x \
                  axis of the histogram.

        """
        if data is None:
            data = numpy.zeros(10)
        data[numpy.isnan(data)] = 0.0
        return numpy.histogram(data, self.n_bins), self.xlim


class Bivariate(StreamingPlot):
    """
    Create a ``holoviews.Bivariate`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "bivariate"

    def __init__(self, data=None, *args, **kwargs):
        """
        Initialize a :class:`Bivariate`.

        Args:
            data: Passed to ``holoviews.Bivariate`.
            *args: Passed to ``holoviews.Bivariate`.
            **kwargs: Passed to ``holoviews.Bivariate`.
        """

        def bivariate(data):
            return holoviews.Bivariate(data, *args, **kwargs)

        super(Bivariate, self).__init__(stream=Pipe, plot=bivariate, data=data)

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        height: int = 350,
        width: int = 400,
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """

        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Bivariate(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                height=height,
                width=width,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                height=height,
                width=width,
                *args,
                **kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )


class Landscape2D(StreamingPlot):
    """
    Plots the interpolated landscaped of values of a set of points.

    The data is visualized creating a :class:`holoviews.QuadMesh` with a \
    :class:`holoviews.Contours` plot with the original data points displayed as \
    a :class:`holoviews.Scatter`.
    """

    name = "landscape"

    def __init__(self, n_points: int = 50, data=None, invert_cmap: bool = False):
        """
        Initialize a :class:`Landscape2d.

        Args:
            n_points: Number of points per dimension used to create the \
                      meshgrid grid that will be used to interpolate the data.
            data: Initial data for the plot.
            invert_cmap: If ``True``, invert the colormap to assign high value \
                         colors to the lowest values.

        """
        self.n_points = n_points
        self.invert_cmap = invert_cmap
        self.xlim = (None, None)
        self.ylim = (None, None)
        super(Landscape2D, self).__init__(stream=Pipe, plot=self.plot_landscape, data=data)

    @staticmethod
    def plot_landscape(data):
        """
        Plot the data as an energy landscape.

        Args:
            data: (x, y, xx, yy, z, xlim, ylim). x, y, z represent the \
                  coordinates of the points that will be interpolated. xx, yy \
                  represent the meshgrid used to interpolate the points. xlim, \
                  ylim are tuples containing the limits of the x and y axes.

        Returns:
            Plot representing the interpolated energy landscape of the target points.

        """
        x, y, xx, yy, z, xlim, ylim = data
        zz = griddata((x, y), z, (xx, yy), method="linear")
        mesh = holoviews.QuadMesh((xx, yy, zz))
        contour = holoviews.operation.contours(mesh, levels=8)
        scatter = holoviews.Scatter((x, y))
        contour_mesh = mesh * contour * scatter
        return contour_mesh.redim(
            x=holoviews.Dimension("x", range=xlim), y=holoviews.Dimension("y", range=ylim),
        )

    def get_plot_data(self, data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]):
        """Create the meshgrid needed to interpolate the target data points."""
        x, y, z = (data[:, 0], data[:, 1], data[:, 2]) if isinstance(data, numpy.ndarray) else data
        # target grid to interpolate to
        xi = numpy.linspace(x.min(), x.max(), self.n_points)
        yi = numpy.linspace(y.min(), y.max(), self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        return x, y, xx, yy, z, self.xlim, self.ylim

    def opts(
        self,
        title="Distribution landscape",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        cmap: str = "default",
        height: int = 350,
        width: int = 350,
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        tools = tools if tools != "default" else ["hover"]
        cmap = cmap if cmap != "default" else ("viridis_r" if self.invert_cmap else "viridis")
        self.plot = self.plot.opts(
            holoviews.opts.QuadMesh(
                cmap=cmap,
                colorbar=True,
                title=title,
                bgcolor="lightgray",
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                height=height,
                width=width,
                *args,
                **kwargs
            ),
            holoviews.opts.Contours(
                cmap=["black"],
                line_width=1,
                alpha=0.9,
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                show_legend=False,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )
