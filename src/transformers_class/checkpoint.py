from pathlib import Path

from flax import nnx, serialization

from .interfaces import LanguageModel


def save_parameters(model: LanguageModel, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = nnx.state(model, nnx.Param)
    pure_parameters = nnx.to_pure_dict(parameters)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_path.write_bytes(serialization.to_bytes(pure_parameters))
    temporary_path.replace(checkpoint_path)


def load_parameters(model: LanguageModel, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' does not exist. Run training first."
        )

    parameter_template = nnx.state(model, nnx.Param)
    pure_template = nnx.to_pure_dict(parameter_template)
    pure_parameters = serialization.from_bytes(
        pure_template,
        checkpoint_path.read_bytes(),
    )
    nnx.replace_by_pure_dict(parameter_template, pure_parameters)
    nnx.update(model, parameter_template)
