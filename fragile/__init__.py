"""Framework for FAI algorithms development."""

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Using or importing the ABCs from 'collections' instead of from 'collections.abc' "
        "is deprecated since Python 3.3,and in 3.9 it will stop working"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "the imp module is deprecated in favour of importlib; see the module's "
        "documentation for alternative uses"
    ),
)
from fragile.core.states import States  # noqa: E402
from fragile.core.walkers import Walkers  # noqa: E402
from fragile.version import __version__  # noqa: E402
