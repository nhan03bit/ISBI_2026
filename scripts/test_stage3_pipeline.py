import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).with_name('stage3_pipeline.py'))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_dependencies_and_recorded_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {key: str(root) for key in ('CKPT', 'TRAIN', 'VAL', 'META')}
            responses = [subprocess.CompletedProcess([], 0, str(i) + '\n', '') for i in (101, 102, 103)]
            with patch.dict(pipeline.os.environ, env), patch.object(pipeline, 'run', side_effect=responses) as execute:
                pipeline.submit(root / 'run', False)
            calls = [call.args[0] for call in execute.call_args_list]
            self.assertNotIn('--dependency', calls[0])
            self.assertIn('afterok:101', calls[1])
            self.assertIn('afterok:102', calls[2])
            self.assertIn('--gpus', calls[1])
            self.assertNotIn('--gpus', calls[0])
            self.assertNotIn('--gpus', calls[2])
            self.assertEqual(json.loads((root / 'run/jobs.json').read_text()),
                             dict(preflight='101', gpu='102', report='103'))
            with self.assertRaises(FileExistsError):
                pipeline.submit(root / 'run', False)

    def test_submission_failure_stops_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {key: str(root) for key in ('CKPT', 'TRAIN', 'VAL', 'META')}
            with patch.dict(pipeline.os.environ, env), patch.object(pipeline, 'run',
                    side_effect=subprocess.CalledProcessError(1, 'sbatch')) as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.submit(root / 'run', False)
            self.assertEqual(execute.call_count, 1)


if __name__ == '__main__':
    unittest.main()
