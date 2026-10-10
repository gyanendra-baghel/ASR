import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from src.prepare_dataset import COUNT, FIELDS, check_manifest, decode_audio, prepare_audio, prepare_dataset, review_page


def recording(text='hello, "world" <script>alert(1)</script>'):
    output = io.BytesIO()
    wave = np.sin(np.arange(4410) * 2 * np.pi * 440 / 44100) * 0.2
    sf.write(output, np.column_stack([wave, wave]), 44100, format='WAV')
    return {'text': text, 'audio_filepath': {'bytes': output.getvalue(), 'path': 'original.wav'}}


class PreparationTests(unittest.TestCase):
    def test_conversion_and_invalid_audio(self):
        samples, rate, extension = decode_audio(recording()['audio_filepath']['bytes'])
        self.assertEqual(extension, 'wav')
        prepared = prepare_audio(samples, rate)
        self.assertEqual(prepared.shape, (1600,))
        self.assertGreater(np.max(np.abs(prepared)), 0.1)
        with self.assertRaises(sf.LibsndfileError):
            decode_audio(b'not audio')
        with self.assertRaises(ValueError):
            decode_audio(self.empty_wav())

    @staticmethod
    def empty_wav():
        output = io.BytesIO()
        sf.write(output, np.empty(0), 16000, format='WAV')
        return output.getvalue()

    def test_pipeline_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('builtins.print'):
                rows = prepare_dataset([recording() for _ in range(COUNT)], root)
                prepare_dataset([recording() for _ in range(COUNT)], root)
            manifest = root / 'dataset/manifests/tiny.csv'
            with manifest.open(newline='', encoding='utf-8') as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(len(loaded), COUNT)
            self.assertEqual(loaded[0]['text'], recording()['text'])
            self.assertEqual(loaded[0]['speaker_id'], '')
            for row in rows:
                info = sf.info(root / row['audio_path'])
                self.assertEqual((info.channels, info.samplerate, info.subtype), (1, 16000, 'PCM_16'))
                self.assertAlmostEqual(row['duration_seconds'], 0.1)
            page = review_page(rows)
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertIn('&lt;script&gt;', page)
            manifest.write_text('unrelated content')
            with self.assertRaises(ValueError):
                prepare_dataset([], root)

    def test_incomplete_run_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'dataset/manifests/tiny.csv'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(','.join(FIELDS) + '\n')
            before = manifest.read_bytes()
            records = [recording(''), {'text':'bad', 'audio_filepath':{'bytes':b'bad'}}, recording()]
            with patch('builtins.print'), self.assertRaises(ValueError):
                prepare_dataset(records, root)
            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(len(list((root / 'dataset/raw/svarah').glob('*.wav'))), 1)

    def test_missing_authentication(self):
        from src.prepare_dataset import main
        with patch('huggingface_hub.get_token', return_value=None), patch('builtins.print') as output:
            self.assertEqual(main(), 1)
        self.assertIn('uv run hf auth login', str(output.call_args))

    def test_network_failure(self):
        from src.prepare_dataset import main
        with patch('huggingface_hub.get_token', return_value='test'), patch('datasets.load_dataset', side_effect=ConnectionError('offline')), patch('builtins.print') as output:
            self.assertEqual(main(), 1)
        self.assertIn('offline', str(output.call_args))


if __name__ == '__main__':
    unittest.main()
