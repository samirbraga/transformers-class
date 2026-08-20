import numpy as np
from pathlib import Path

from .utils import download_file, process_corpus
from .typings import TokenCorpus


DATA_DIR: Path = Path("datasets/tiny_shakespeare")
TINY_SHAKESPEARE_URL: str = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
CORPUS_FILE: Path = DATA_DIR / "tinyshakespeare.txt"


def load_tiny_shakespeare() -> tuple[
    str,
    list[tuple[str, str]],
    dict[str, int],
    set[str],
]:
    content = download_file(TINY_SHAKESPEARE_URL, CORPUS_FILE)
    turns, vocabulary, speakers = process_corpus(CORPUS_FILE)
    return content, turns, vocabulary, speakers


def load_corpus() -> tuple[
    TokenCorpus,
    TokenCorpus,
    dict[str, int],
    dict[int, str],
]:
    text, _, id_to_char, _ = load_tiny_shakespeare()
    char_to_id = {char: idx for idx, char in id_to_char.items()}
    encoded = np.asarray([char_to_id[character] for character in text], dtype=np.int32)
    split_index = int(0.9 * len(encoded))
    return encoded[:split_index], encoded[split_index:], char_to_id, id_to_char

def main() -> None:
    _, _, _, speakers = load_tiny_shakespeare()
    print(speakers)


if __name__ == "__main__":
    main()
