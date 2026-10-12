"""Character tokenization without transcript normalization.

ID 0 is the CTC blank. Batch padding is not a vocabulary token: a future
collator can use PAD_ID and must exclude padding using transcript lengths.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PAD_ID = -100


class CharacterTokenizer:
    blank_id = 0

    def __init__(self, characters):
        if not isinstance(characters, list) or any(not isinstance(c, str) or len(c) != 1 for c in characters):
            raise ValueError("Vocabulary must be a list of single Unicode characters")
        if characters != sorted(set(characters)):
            raise ValueError("Vocabulary characters must be unique and sorted")
        self.characters = tuple(characters)
        self._character_to_id = {character: index + 1 for index, character in enumerate(characters)}

    @property
    def vocab_size(self):
        return len(self.characters) + 1

    @classmethod
    def from_texts(cls, texts):
        characters = set()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Transcripts must be strings")
            characters.update(text)
        if not characters:
            raise ValueError("Cannot build a vocabulary from empty transcripts")
        return cls(sorted(characters))

    def encode(self, text):
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        result = []
        for index, character in enumerate(text):
            if character not in self._character_to_id:
                raise ValueError(f"Unknown character {character!r} at position {index}; rebuild the vocabulary from training transcripts")
            result.append(self._character_to_id[character])
        return result

    def _validate_id(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise ValueError(f"Invalid token ID: {token!r}")

    def decode(self, tokens):
        characters = []
        for token in tokens:
            self._validate_id(token)
            if token == self.blank_id:
                raise ValueError("CTC blank is not a text character; use decode_ctc for model predictions")
            characters.append(self.characters[token - 1])
        return ''.join(characters)

    def decode_ctc(self, tokens):
        """Collapse consecutive predictions first, then remove blank tokens."""
        collapsed = []
        previous = None
        for token in tokens:
            self._validate_id(token)
            if token != previous and token != self.blank_id:
                collapsed.append(token)
            previous = token
        return self.decode(collapsed)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=1, blank_id=self.blank_id, characters=list(self.characters))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or type(payload.get('version')) is not int or payload['version'] != 1 or type(payload.get('blank_id')) is not int or payload['blank_id'] != 0:
            raise ValueError("Unsupported tokenizer format or CTC blank ID")
        return cls(payload.get('characters'))


def read_transcripts(manifest):
    with Path(manifest).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'text' not in reader.fieldnames:
            raise ValueError("Manifest must have a text column")
        texts = []
        for line, row in enumerate(reader, start=2):
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or missing transcript at manifest row {line}")
            texts.append(text)
    if not texts:
        raise ValueError("Manifest contains no transcripts")
    return texts


def build_tokenizer(manifest, output):
    texts = read_transcripts(manifest)
    tokenizer = CharacterTokenizer.from_texts(texts)
    for index, text in enumerate(texts, start=1):
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise ValueError(f"Transcript {index} failed the round-trip check")
    tokenizer.save(output)
    restored = CharacterTokenizer.load(output)
    for text in texts:
        if restored.decode(restored.encode(text)) != text:
            raise ValueError("Saved tokenizer failed the round-trip check")
    return restored, len(texts)


def main():
    parser = argparse.ArgumentParser(description='Build and validate an exact character tokenizer from a training manifest')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'dataset/manifests/tiny.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tokenizer.json')
    args = parser.parse_args()
    try:
        tokenizer, count = build_tokenizer(args.manifest, args.output)
    except (OSError, ValueError) as error:
        print(f'Tokenizer creation failed: {error}', file=sys.stderr)
        return 1
    print(f'Validated {count} exact transcript round trips.\nVocabulary size: {tokenizer.vocab_size} (including CTC blank ID 0)\nSaved tokenizer: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
