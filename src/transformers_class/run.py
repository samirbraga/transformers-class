import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import optax
from flax import nnx

from .generation import generate_with_top_k_sampling
from .interfaces import LanguageModel
from .load_dataset import load_corpus
from .mlp import MLPLanguageModel
from .train import train
from .transformer import TransformerLanguageModel
from .typings import TokenCorpus


CONTEXT_LENGTH: int = 64
BATCH_SIZE: int = 512
TRAINING_STEPS: int = 20_000
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-3
DROPOUT_RATE: float = 0.1
EVAL_INTERVAL: int = 500
GENERATION_TEMPERATURE: float = 0.7
GENERATION_TOP_K: int = 12

MLP_EMBEDDING_DIM: int = 32
MLP_HIDDEN_DIM: int = 256

TRANSFORMER_D_MODEL: int = 128
TRANSFORMER_NUM_HEADS: int = 4
TRANSFORMER_NUM_LAYERS: int = 2
TRANSFORMER_D_FF: int = 512


def parameter_count(model: LanguageModel) -> int:
    parameters = nnx.state(model, nnx.Param)
    return sum(array.size for array in jax.tree.leaves(parameters))


def train_model[ModelT: LanguageModel](
    name: str,
    model: ModelT,
    train_data: TokenCorpus,
    validation_data: TokenCorpus,
) -> tuple[ModelT, list[float], list[float]]:
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        wrt=nnx.Param,
    )

    print(f"\n--- Training {name} ---")
    print(f"Parameters: {parameter_count(model):,}")

    return train(
        model,
        optimizer,
        TRAINING_STEPS,
        BATCH_SIZE,
        EVAL_INTERVAL,
        train_data,
        validation_data,
    )


def evaluation_steps(history_length: int) -> list[int]:
    steps = [
        step
        for step in range(1, TRAINING_STEPS + 1)
        if step == 1 or step % EVAL_INTERVAL == 0
    ]
    return steps[:history_length]


def plot_losses(
    mlp_training_loss: list[float],
    mlp_validation_loss: list[float],
    transformer_training_loss: list[float],
    transformer_validation_loss: list[float],
) -> None:
    mlp_steps = evaluation_steps(len(mlp_training_loss))
    transformer_steps = evaluation_steps(len(transformer_training_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(
        mlp_steps,
        mlp_training_loss,
        color="tab:blue",
        linestyle="--",
        label="MLP training",
    )
    plt.plot(
        mlp_steps,
        mlp_validation_loss,
        color="tab:blue",
        label="MLP validation",
    )
    plt.plot(
        transformer_steps,
        transformer_training_loss,
        color="tab:orange",
        linestyle="--",
        label="Transformer training",
    )
    plt.plot(
        transformer_steps,
        transformer_validation_loss,
        color="tab:orange",
        label="Transformer validation",
    )
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLP vs Transformer on Tiny Shakespeare")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150)
    plt.close()


def print_generation(
    name: str,
    model: LanguageModel,
    validation_data: TokenCorpus,
    id_to_char: dict[int, str],
    seed: int,
    output_file: Path,
) -> None:
    initial_tokens = validation_data[: model.context_length]
    generated_text = generate_with_top_k_sampling(
        model,
        initial_tokens,
        id_to_char,
        model.context_length,
        jax.random.key(seed),
        temperature=GENERATION_TEMPERATURE,
        top_k=GENERATION_TOP_K,
    )
    output_file.write_text(generated_text, encoding="utf-8")
    print(f"\n--- {name} generated text ---\n")
    print(generated_text)
    print(f"\nGenerated text saved to {output_file}")


def main() -> None:
    train_data, validation_data, char_to_id, id_to_char = load_corpus()
    print(f"Vocabulary size: {len(char_to_id)}")
    print(f"JAX devices: {jax.devices()}")

    mlp = MLPLanguageModel(
        vocab_size=len(char_to_id),
        embedding_dim=MLP_EMBEDDING_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        context_length=CONTEXT_LENGTH,
        dropout_rate=DROPOUT_RATE,
        rngs=nnx.Rngs(42),
    )
    transformer = TransformerLanguageModel(
        vocab_size=len(char_to_id),
        context_length=CONTEXT_LENGTH,
        d_model=TRANSFORMER_D_MODEL,
        num_heads=TRANSFORMER_NUM_HEADS,
        num_layers=TRANSFORMER_NUM_LAYERS,
        d_ff=TRANSFORMER_D_FF,
        dropout_rate=DROPOUT_RATE,
        rngs=nnx.Rngs(43),
    )

    mlp, mlp_training_loss, mlp_validation_loss = train_model(
        "MLP",
        mlp,
        train_data,
        validation_data,
    )
    transformer, transformer_training_loss, transformer_validation_loss = (
        train_model(
            "Transformer",
            transformer,
            train_data,
            validation_data,
        )
    )

    plot_losses(
        mlp_training_loss,
        mlp_validation_loss,
        transformer_training_loss,
        transformer_validation_loss,
    )
    print("\nLoss plot saved to loss_comparison.png")

    print_generation(
        "MLP",
        mlp,
        validation_data,
        id_to_char,
        seed=44,
        output_file=Path("mlp_generated.txt"),
    )
    print_generation(
        "Transformer",
        transformer,
        validation_data,
        id_to_char,
        seed=44,
        output_file=Path("transformer_generated.txt"),
    )


if __name__ == "__main__":
    main()
