import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # 3. Path field-ləri
    raw_data_dir: Path = field(default_factory=lambda: Path("artifacts/raw_data"))
    processed_data_dir: Path = field(default_factory=lambda: Path("artifacts/data"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("artifacts/checkpoints"))
    tokenizer_dir: Path = field(default_factory=lambda: Path("artifacts/tokenizers"))

    # 4. Data parametrləri
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 60
    vocab_size: int = 16000
    num_workers: int = 0  # Windows üçün 0

    # 5. Model parametrləri
    embedding_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.35
    bidirectional: bool = True

    # 6. Training parametrləri
    batch_size: int = 64
    learning_rate: float = 5e-4
    max_epochs: int = 15
    early_stopping_patience: int = 5
    teacher_forcing_ratio: float = 0.5
    gradient_clip_val: float = 1.0
    wandb_project: str = "rnn-en-es-translation"
    wandb_run_name: str = "gru-bahdanau-baseline"

    def __post_init__(self) -> None:
        # String ötürülərsə Path obyektinə çevrilməsini təmin etmək
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.tokenizer_dir = Path(self.tokenizer_dir)

        # Hər bir split ratio-nun [0.0, 1.0] aralığında olmasını yoxlamaq
        for name, ratio in [
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if not (0.0 <= ratio <= 1.0):
                raise ValueError(
                    f"{name} [0.0, 1.0] aralığında olmalıdır! Daxil edilən: {ratio}"
                )

        # Split nisbətlərinin cəmi 1.0 olmalıdır (abs_tol ilə float dəqiqliyi)
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total_ratio, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"train_ratio, val_ratio və test_ratio cəmi 1.0 olmalıdır! Hazırkı cəm: {total_ratio}"
            )

        # Əlavə parametr validation-ları
        if self.max_length <= 0:
            raise ValueError(
                f"max_length 0-dan böyük olmalıdır! Daxil edilən: {self.max_length}"
            )

        if self.vocab_size <= 4:
            raise ValueError(
                f"vocab_size 4-dən böyük olmalıdır! Daxil edilən: {self.vocab_size}"
            )

        if self.num_workers < 0:
            raise ValueError(
                f"num_workers 0 və ya daha böyük olmalıdır! Daxil edilən: {self.num_workers}"
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim 0-dan böyük olmalıdır! Daxil edilən: {self.embedding_dim}"
            )

        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size 0-dan böyük olmalıdır! Daxil edilən: {self.hidden_size}"
            )

        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers 0-dan böyük olmalıdır! Daxil edilən: {self.num_layers}"
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout [0, 1) aralığında olmalıdır! Daxil edilən: {self.dropout}"
            )

        if not (0.0 <= self.teacher_forcing_ratio <= 1.0):
            raise ValueError(
                f"teacher_forcing_ratio [0, 1] aralığında olmalıdır! Daxil edilən: {self.teacher_forcing_ratio}"
            )

        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size 0-dan böyük olmalıdır! Daxil edilən: {self.batch_size}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate 0-dan böyük olmalıdır! Daxil edilən: {self.learning_rate}"
            )

        if self.max_epochs <= 0:
            raise ValueError(
                f"max_epochs 0-dan böyük olmalıdır! Daxil edilən: {self.max_epochs}"
            )

        if self.gradient_clip_val < 0:
            raise ValueError(
                f"gradient_clip_val mənfi ola bilməz! Daxil edilən: {self.gradient_clip_val}"
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience mənfi ola bilməz! Daxil edilən: {self.early_stopping_patience}"
            )
        if not self.wandb_project.strip():
            raise ValueError("wandb_project boş ola bilməz.")

        if not self.wandb_run_name.strip():
            raise ValueError("wandb_run_name boş ola bilməz.")


if __name__ == "__main__":
    print("Default config testi...")
    config = TrainingConfig()
    print("✓ Default config uğurla yarandı!")

    # String path-in Path obyektinə çevrilməsi testi
    str_config = TrainingConfig(raw_data_dir="custom/path")
    assert isinstance(str_config.raw_data_dir, Path), (
        "String path Path obyektinə çevrilmədi!"
    )
    print("✓ String path -> Path konversiyası uğurla keçdi!")

    # Xəta sınaqları (Edge cases)
    test_cases = [
        (
            "Mənfi split ratio",
            {"train_ratio": -0.2, "val_ratio": 0.6, "test_ratio": 0.6},
        ),
        ("num_layers = 0", {"num_layers": 0}),
        ("num_workers = -1", {"num_workers": -1}),
        ("batch_size = 0", {"batch_size": 0}),
        ("dropout = 1.0", {"dropout": 1.0}),
        ("teacher_forcing_ratio = 1.5", {"teacher_forcing_ratio": 1.5}),
    ]

    for name, kwargs in test_cases:
        try:
            TrainingConfig(**kwargs)
            print(f"✗ XƏTA: {name} üçün xəta tutulmadı!")
        except ValueError as e:
            print(f"✓ {name} uğurla tutuldu: {e}")
