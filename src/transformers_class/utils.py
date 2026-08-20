from pathlib import Path
import requests


def download_file(url: str, filename: Path | str):
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists():
        print(f"File '{filename}' already exists. Skipping download.")
        return filename.read_text(encoding="utf-8")

    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    return response.content

def index_vocabulary(vocabulary: set[str]) -> dict[str, int]:
    return {char: idx for idx, char in enumerate(sorted(vocabulary))}

def process_corpus(dataset_path: Path | str) -> tuple[list[tuple[str, str]],  dict[str, int], set[str]]:
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file '{dataset_path}' does not exist.")

    with open(dataset_path, "r", encoding="utf-8") as file:
        content = file.read()

    vocabulary = set(content)

    turns = content.split("\n\n")
    turns = [tuple(turn.split(":\n", 1)) for turn in turns if len(turn.split(":\n", 1)) > 1]
    speakers = set(speaker for (speaker, _) in turns)
    
    return turns, index_vocabulary(vocabulary), speakers
