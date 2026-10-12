import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.tokenizer import CharacterTokenizer, PAD_ID, build_tokenizer, read_transcripts


class TokenizerTests(unittest.TestCase):
    def test_exact_round_trip_and_deterministic_ids(self):
        texts = ['hello  book!', 'Hindi: नमस्ते', 'UPI, "Pay" 123.']
        tokenizer = CharacterTokenizer.from_texts(texts)
        reverse = CharacterTokenizer.from_texts(reversed(texts))
        self.assertEqual(tokenizer.characters, reverse.characters)
        for text in texts:
            tokens = tokenizer.encode(text)
            self.assertNotIn(tokenizer.blank_id, tokens)
            self.assertNotIn(PAD_ID, tokens)
            self.assertEqual(tokenizer.decode(tokens), text)
        self.assertEqual(tokenizer.decode(tokenizer.encode('')), '')

    def test_ctc_repeats_separated_by_blanks(self):
        tokenizer = CharacterTokenizer.from_texts(['book'])
        b, o, k = tokenizer.encode('bok')
        self.assertEqual(tokenizer.decode_ctc([0,b,b,o,o,0,o,o,k,k,0]), 'book')
        self.assertEqual(tokenizer.decode_ctc([0,0]), '')
        self.assertEqual(tokenizer.decode([o,o]), 'oo')

    def test_invalid_input(self):
        tokenizer = CharacterTokenizer.from_texts(['a'])
        with self.assertRaises(ValueError):
            tokenizer.encode('b')
        for token in [-100, -1, 2, True, 1.0]:
            with self.assertRaises(ValueError):
                tokenizer.decode([token])
        with self.assertRaises(ValueError):
            tokenizer.decode([0])
        with self.assertRaises(ValueError):
            CharacterTokenizer(['a','a'])

    def test_build_save_load_csv_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'tiny.csv'
            text = 'hello, "book" नमस्ते'
            with manifest.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['text'])
                writer.writerow([text])
            tokenizer, count = build_tokenizer(manifest, root / 'tokenizer.json')
            self.assertEqual(count, 1)
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            payload = json.loads((root / 'tokenizer.json').read_text())
            payload['blank_id'] = 1
            (root / 'tokenizer.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                CharacterTokenizer.load(root / 'tokenizer.json')

    def test_invalid_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'tiny.csv'
            for content in ['audio_path\nclip.wav\n', 'text\n', 'text\n""\n']:
                manifest.write_text(content)
                with self.assertRaises(ValueError):
                    read_transcripts(manifest)


if __name__ == '__main__':
    unittest.main()
