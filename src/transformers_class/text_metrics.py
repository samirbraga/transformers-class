import re
from dataclasses import asdict, dataclass


WORD_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ANSI_RED: str = "\033[31m"
ANSI_RESET: str = "\033[0m"


@dataclass(frozen=True)
class TextMetrics:
    word_count: int
    corpus_word_count: int
    corpus_word_rate: float
    unique_word_count: int
    unique_word_rate: float
    average_word_length: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def extract_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def build_word_vocabulary(reference_text: str) -> set[str]:
    return set(extract_words(reference_text))


def highlight_words_outside_vocabulary(
    text: str,
    word_vocabulary: set[str],
) -> str:
    def highlight(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() in word_vocabulary:
            return word
        return f"{ANSI_RED}{word}{ANSI_RESET}"

    return WORD_PATTERN.sub(highlight, text)


def evaluate_generated_text(
    generated_text: str,
    word_vocabulary: set[str],
) -> TextMetrics:
    words = extract_words(generated_text)
    if not words:
        return TextMetrics(0, 0, 0.0, 0, 0.0, 0.0)

    corpus_word_count = sum(word in word_vocabulary for word in words)
    unique_word_count = len(set(words))
    return TextMetrics(
        word_count=len(words),
        corpus_word_count=corpus_word_count,
        corpus_word_rate=corpus_word_count / len(words),
        unique_word_count=unique_word_count,
        unique_word_rate=unique_word_count / len(words),
        average_word_length=sum(map(len, words)) / len(words),
    )
