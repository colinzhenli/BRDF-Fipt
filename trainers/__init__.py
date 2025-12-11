from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer

# Registry for easy lookup
TRAINER_REGISTRY = {
    1: Stage1Trainer,
    2: Stage2Trainer,
}

def get_trainer_class(stage: int):
    """Factory function to get trainer class by stage."""
    if stage not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[stage]