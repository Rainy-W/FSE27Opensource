from pathlib import Path
import tempfile
import unittest

from humer.config import HumerConfig, SUPPORTED_MODELS


class ConfigTests(unittest.TestCase):
    def test_release_supports_only_paper_models(self) -> None:
        self.assertEqual(
            SUPPORTED_MODELS,
            ("codebert", "unixcoder", "codet5", "linevul", "vulgpt", "epvd"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = HumerConfig(
                data_dir=str(root), output_dir=str(root / "out"), model="unixcoder"
            )
            config.validate()
            self.assertEqual(config.model_name, "microsoft/unixcoder-base")

    def test_consolidation_cannot_be_disabled(self) -> None:
        config = HumerConfig(
            data_dir="data",
            output_dir="output",
            model="codebert",
            consolidation_max_epochs=0,
        )
        with self.assertRaisesRegex(ValueError, "cannot be disabled"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
