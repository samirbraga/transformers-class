import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE

from .typings import TokenCorpus


DATASET_NAME: str = "roneneldan/TinyStories"
UNK_TOKEN: str = "<|unk|>"
BOS_TOKEN: str = "<|beginofstory|>"
EOS_TOKEN: str = "<|endofstory|>"


def _cache_stories(split: str, limit: int, output_file: Path) -> None:
    if output_file.exists():
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    written = 0
    with temporary_file.open("w", encoding="utf-8") as file:
        for row in dataset:
            text = str(row["text"]).strip()
            if not text:
                continue
            file.write(json.dumps(text, ensure_ascii=False) + "\n")
            written += 1
            if written >= limit:
                break

    if written < limit:
        temporary_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"TinyStories split '{split}' yielded {written} non-empty stories; "
            f"{limit} were requested."
        )
    temporary_file.replace(output_file)


def iter_stories(stories_file: Path) -> Iterator[str]:
    with stories_file.open("r", encoding="utf-8") as file:
        for line in file:
            yield str(json.loads(line))


def train_bpe_tokenizer(
    stories_file: Path,
    tokenizer_file: Path,
    vocab_size: int,
    story_count: int,
) -> Tokenizer:
    if tokenizer_file.exists():
        return Tokenizer.from_file(str(tokenizer_file))

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        iter_stories(stories_file),
        trainer=trainer,
        length=story_count,
    )
    tokenizer_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_file), pretty=True)
    return tokenizer


def encode_stories(
    stories_file: Path,
    tokenizer: Tokenizer,
    encoded_file: Path,
) -> TokenCorpus:
    if encoded_file.exists():
        return np.load(encoded_file)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("The TinyStories tokenizer is missing BOS or EOS tokens")

    token_ids: list[int] = []
    for story in iter_stories(stories_file):
        token_ids.append(bos_id)
        token_ids.extend(tokenizer.encode(story, add_special_tokens=False).ids)
        token_ids.append(eos_id)

    encoded = np.asarray(token_ids, dtype=np.int32)
    encoded_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(encoded_file, encoded)
    return encoded


def prepare_tinystories(
    data_dir: Path = Path("datasets/tinystories"),
    train_stories: int = 20_000,
    validation_stories: int = 2_000,
    vocab_size: int = 2_048,
) -> tuple[TokenCorpus, TokenCorpus, Tokenizer, str]:
    if train_stories < 1 or validation_stories < 1:
        raise ValueError("Story counts must be positive")
    if vocab_size < 512:
        raise ValueError("vocab_size must be at least 512 for byte-level BPE")

    cache_dir = data_dir / (
        f"train_{train_stories}_validation_{validation_stories}_vocab_{vocab_size}"
    )
    train_file = cache_dir / "train.jsonl"
    validation_file = cache_dir / "validation.jsonl"
    tokenizer_file = cache_dir / "tokenizer.json"

    _cache_stories("train", train_stories, train_file)
    _cache_stories("validation", validation_stories, validation_file)
    tokenizer = train_bpe_tokenizer(
        train_file,
        tokenizer_file,
        vocab_size,
        train_stories,
    )
    train_tokens = encode_stories(
        train_file,
        tokenizer,
        cache_dir / "train_tokens.npy",
    )
    validation_tokens = encode_stories(
        validation_file,
        tokenizer,
        cache_dir / "validation_tokens.npy",
    )
    reference_text = "\n".join(iter_stories(train_file))
    return train_tokens, validation_tokens, tokenizer, reference_text
