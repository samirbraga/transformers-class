import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .interfaces import LanguageModel
from .typings import Scalar, TokenBatch, TokenCorpus, TokenTargets




def sample_batch(
    data: TokenCorpus,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
) -> tuple[TokenBatch, TokenTargets]:
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    inputs = np.stack([data[start : start + context_length] for start in starts])
    targets = data[starts + context_length]
    return jnp.asarray(inputs), jnp.asarray(targets)


def loss_fn(
    model: LanguageModel,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    logits = model(inputs, deterministic=False)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


@nnx.jit
def train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    loss, gradients = nnx.value_and_grad(loss_fn)(model, inputs, targets)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def evaluate(
    model: LanguageModel,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    logits = model(inputs, deterministic=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def train[ModelT: LanguageModel](
    model: ModelT,
    optimizer: nnx.Optimizer,
    training_steps: int,
    batch_size: int,
    eval_interval: int,
    train_data: TokenCorpus,
    validation_data: TokenCorpus,
) -> tuple[ModelT, list[float], list[float]]:

    training_loss_history = []
    validation_loss_history = []

    rng = np.random.default_rng(42)
    for step in range(1, training_steps + 1):
        inputs, targets = sample_batch(train_data, batch_size, model.context_length, rng)
        training_loss = train_step(model, optimizer, inputs, targets)

        if step == 1 or step % eval_interval == 0:
            validation_inputs, validation_targets = sample_batch(
                validation_data,
                batch_size * 2,
                model.context_length,
                rng,
            )
            validation_loss = evaluate(model, validation_inputs, validation_targets)

            validation_loss_history.append(float(validation_loss))
            training_loss_history.append(float(training_loss))

            print(
                f"step={step:5d} "
                f"train_loss={float(training_loss):.4f} "
                f"validation_loss={float(validation_loss):.4f} "
                f"perplexity={float(jnp.exp(validation_loss)):.2f}"
            )

    return model, training_loss_history, validation_loss_history
