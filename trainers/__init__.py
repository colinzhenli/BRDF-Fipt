from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer
from .stage1_trainer_merl import Stage1Trainer_MERL

# Registry for easy lookup
TRAINER_REGISTRY = {
    1: Stage1Trainer_MERL,
    2: Stage1Trainer_MERL,
}

def get_trainer_class(stage: int):
    """Factory function to get trainer class by stage."""
    if stage not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[stage]