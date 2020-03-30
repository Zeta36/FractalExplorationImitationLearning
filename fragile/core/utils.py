import copy
from typing import Any, Dict, Generator, Tuple, Union

import numpy
from PIL import Image

try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(*args, **kwargs):
        """If not using jupyter notebook do nothing."""
        pass


RANDOM_SEED = 160290
random_state = numpy.random.RandomState(seed=RANDOM_SEED)

float_type = numpy.float32
Scalar = Union[int, numpy.int, float, numpy.float]
StateDict = Dict[str, Dict[str, Any]]

NUMPY_IGNORE_WARNINGS_PARAMS = {
    "divide": "ignore",
    "over": "ignore",
    "under": "ignore",
    "invalid": "ignore",
}


def remove_notebook_margin(output_width_pct: int = 80):
    """Make the notebook output wider."""
    from IPython.core.display import HTML

    html = (
        "<style>"
        ".container { width:" + str(output_width_pct) + "% !important; }"
        ".input{ width:70% !important; }"
        ".text_cell{ width:70% !important;"
        " font-size: 16px;}"
        ".title {align:center !important;}"
        "</style>"
    )
    return HTML(html)


def hash_numpy(x: numpy.ndarray) -> int:
    """Return a value that uniquely identifies a numpy array."""
    return hash(x.tostring())


def resize_frame(
    frame: numpy.ndarray, height: int, width: int, mode: str = "RGB"
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        height: Height of the resized image.
        width: Width of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize((height, width))
    return numpy.array(frame)


def params_to_tensors(param_dict, n_walkers: int):
    """Transform a parameter dictionary into an array dictionary."""
    tensor_dict = {}
    copy_dict = copy.deepcopy(param_dict)
    for key, val in copy_dict.items():
        sizes = tuple([n_walkers]) + val["size"]
        del val["size"]
        tensor_dict[key] = numpy.empty(sizes, **val)
    return tensor_dict


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return x.mean(), x.std(), x.max(), x.min()
    except AttributeError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


def similiar_chunks_indexes(n_values, n_chunks) -> Generator[Tuple[int, int], None, None]:
    """
    Return the indexes for splitting an array in similar chunks.

    Args:
        n_values: Length of the array that will be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator containing the indexes of every new chunk.

    """
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for i in range(0, n_values, chunk_size):
        yield i, i + chunk_size


def split_similar_chunks(
    vector: Union[list, numpy.ndarray], n_chunks: int
) -> Generator[Union[list, numpy.ndarray], None, None]:
    """
    Split an indexable object into similar chunks.

    Args:
        vector: Target object to be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator that returns the chunks created after splitting the target object.

    """
    for start, end in similiar_chunks_indexes(len(vector), n_chunks):
        yield vector[start:end]
