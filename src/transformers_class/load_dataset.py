from pathlib import Path
from src.transformers_class.utils import download_file, process_corpus

data_dir = Path('./datasets/tiny_shakespeare')

tinyshakespeare_file = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

corpus_file = data_dir / 'tinyshakespeare.txt'
content = download_file(tinyshakespeare_file, corpus_file)

turns, vocabulary, speakers = process_corpus(corpus_file)

print(speakers)