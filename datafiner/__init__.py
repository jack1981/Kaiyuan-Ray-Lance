"""Public node exports for YAML-driven Ray Data pipeline construction.

Importing this package eagerly registers most built-in node classes and
best-effort imports optional nodes whose third-party dependencies may be absent.
The exported operators cover the same filtering, deduplication, sampling, and
mixing primitives emphasized by the PCMind-2.1 preprocessing framework.
See also `datafiner/register.py` for the registration mechanism.
"""

from .data_reader import (
    LanceReader,
    LanceReaderZstd,
    JsonlZstReader,
    JsonReader,
    NpyReader,
    FormatReader,
)
from .data_writer import LanceWriter, LanceWriterZstd
from .common import (
    ColumnAlias,
    ColumnDrop,
    ColumnSelect,
    Schema,
    RowNumber,
    Stat,
)
from .visualizer import Visualizer
from .filter import Filter
from .shuffler import Shuffler
from .transform import (
    AddConstants,
    ConversationToParagraph,
    ConcatenateColumns,
)
from .selector import Selector
from .join import Joiner
from .counter import TokenCounter_v2
from .reorder import Reorder
from .quantile import AddRankQuantile
from .group_reorder import InterleavedReorder
from .union import UnionByPosition, UnionByName
from .concat import Concat
from .splitter import Splitter
from .sampler import Sampler, GroupFlatten, Flatten

try:
    from .deduplication.minhash import MinHash
except ModuleNotFoundError:
    pass

try:
    from .tokenization import Tokenization
except ModuleNotFoundError:
    pass

try:
    from .detokenization import Detokenization, BatchDetokenization
except ModuleNotFoundError:
    pass

try:
    from .text_scorer import FastTextScorer, FastTextFilter, SeqClassifierScorer
except ModuleNotFoundError:
    pass

# Optional modules: keep package importable when these files are absent.
try:
    from .summary import TokenCounter
except ModuleNotFoundError:
    pass

try:
    from .rescale_score import ScoreRescaler
except ModuleNotFoundError:
    pass
