from typing import Protocol

from .typings import Logits, TokenBatch


class LanguageModel(Protocol):
    """Common interface for next-token language models."""

    context_length: int

    def __call__(
        self,
        tokens: TokenBatch,
        *,
        deterministic: bool,
    ) -> Logits: ...
