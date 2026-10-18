"""Check that the independent audit catches a corrupted saved trajectory."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coordinate_geometry.part1_experiments import main
from coordinate_geometry.verify_part1 import verify


class Part1ArtifactTests(unittest.TestCase):
    def test_generated_artifacts_pass_and_corruption_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                main(["--config", str(root / "configs/part1.json"),
                      "--output", str(output), "--no-plots"])
            report = verify(output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["trajectories"], 56)
            path = output / "trajectories.json"
            saved = json.loads(path.read_text())
            saved["cases"][0]["runs"][0]["positions"][1][0] += 0.01
            path.write_text(json.dumps(saved))
            with self.assertRaises(AssertionError):
                verify(output)


if __name__ == "__main__":
    unittest.main()
