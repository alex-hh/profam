from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("profam")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.1.8"

from profam.api import GenerationResult, ProFam, ScoringResult

__all__ = ["ProFam", "GenerationResult", "ScoringResult", "__version__"]
