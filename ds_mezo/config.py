from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ModelConfig:
    path: str = MISSING
    adapter_path: str = ""
    rank: int = 16
    targets: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    total_steps: int = 1000
    seed: int = 42
    resume_from: str = ""


@dataclass
class EvalConfig:
    n_samples: int = 20
    temperature: float = 0.2
    eval_at_steps: list = field(default_factory=list)


@dataclass
class DataConfig:
    train_data: str = "mbpp"


@dataclass
class SystemConfig:
    staging_dir: str = "/dev/shm/ds_mezo"
    lock_clocks: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    output_dir: str = MISSING
    dsmezo_results: str = ""
