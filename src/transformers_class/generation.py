import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .interfaces import LanguageModel
from .typings import TokenCorpus


def generate_with_multinomial_sampling(
    model: LanguageModel,
    initial_tokens: TokenCorpus,
    id_to_char: dict[int, str],
    context_length: int,
    key: PRNGKeyArray,
    length: int = 500,
    temperature: float = 0.8,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    tokens = initial_tokens.tolist()

    for _ in range(length):
        context = jnp.asarray(tokens[-context_length:], dtype=jnp.int32)[None, :]
        logits = model(context, deterministic=True)[0]
        key, sample_key = jax.random.split(key)
        next_token = jax.random.categorical(sample_key, logits / temperature)
        tokens.append(int(next_token))

    return "".join(id_to_char[token] for token in tokens)


def generate_with_top_k_sampling(
    model: LanguageModel,
    initial_tokens: TokenCorpus,
    id_to_char: dict[int, str],
    context_length: int,
    key: PRNGKeyArray,
    length: int = 500,
    temperature: float = 0.7,
    top_k: int = 12,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    tokens = initial_tokens.tolist()

    for _ in range(length):
        context = jnp.asarray(tokens[-context_length:], dtype=jnp.int32)[None, :]
        logits = model(context, deterministic=True)[0] / temperature
        number_of_candidates = min(top_k, logits.shape[-1])
        top_logits, top_token_ids = jax.lax.top_k(logits, number_of_candidates)

        key, sample_key = jax.random.split(key)
        candidate_index = jax.random.categorical(sample_key, top_logits)
        next_token = top_token_ids[candidate_index]
        tokens.append(int(next_token))

    return "".join(id_to_char[token] for token in tokens)
