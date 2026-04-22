"""Transcoder implementations and training."""

from .per_layer import (
    PerLayerTranscoder,
    TranscoderConfig,
    TranscoderOutput,
    jumprelu,
)
from .cross_layer import (
    CrossLayerTranscoder,
    CLTConfig,
    CLTOutput,
)
from .train import (
    TrainingConfig,
    TranscoderTrainer,
    MLPActivationDataset,
    evaluate_transcoder,
    CLTTrainingConfig,
    CLTTrainer,
    CrossLayerDataset,
    evaluate_clt,
)

__all__ = [
    # Per-layer transcoder
    "PerLayerTranscoder",
    "TranscoderConfig",
    "TranscoderOutput",
    "jumprelu",
    # Cross-layer transcoder
    "CrossLayerTranscoder",
    "CLTConfig",
    "CLTOutput",
    # Training
    "TrainingConfig",
    "TranscoderTrainer",
    "MLPActivationDataset",
    "evaluate_transcoder",
    "CLTTrainingConfig",
    "CLTTrainer",
    "CrossLayerDataset",
    "evaluate_clt",
]