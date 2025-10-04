"""CASCADE Collector Package - Main entry point for the data collection system."""

__version__ = "0.1.0"

from .collectors import (
    KiwiClient,
    Recorder,
    WebSDRClient,
    HybridSDRSelector,
    SDRManager,
    EventScaler,
    CollectionWorker,
    QueueManager
)

from .processors import (
    FT8Processor,
    WSPRProcessor,
    QRNProcessor,
    SignalExtractor
)

from .storage import (
    TigrisStorage,
    FileManager,
    MetadataStore
)

__all__ = [
    # Collectors
    "KiwiClient",
    "Recorder",
    "WebSDRClient",
    "HybridSDRSelector",
    "SDRManager",
    "EventScaler",
    "CollectionWorker",
    "QueueManager",

    # Processors
    "FT8Processor",
    "WSPRProcessor",
    "QRNProcessor",
    "SignalExtractor",

    # Storage
    "TigrisStorage",
    "FileManager",
    "MetadataStore"
]