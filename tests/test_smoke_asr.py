import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.smoke_asr import run_command


class SmokeDiagnosticsTests(unittest.TestCase):
    def test_failed_cli_keeps_structured_stdout_error_and_stderr_progress(self):
        script = ("import json, sys; print(json.dumps({'results': [{'status': 'error', "
                  "'error': 'Synthetic model failure'}]})); "
                  "print('Synthetic progress', file=sys.stderr); sys.exit(1)")
        with self.assertRaises(RuntimeError) as raised:
            run_command([sys.executable, "-c", script])
        self.assertIn("Synthetic model failure", str(raised.exception))
        self.assertIn("Synthetic progress", str(raised.exception))

    def test_command_logs_are_retained_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            for code in [0, 1]:
                prefix = Path(temporary) / f"command-{code}"
                script = (f"import json, sys; print(json.dumps({{'value': 42}})); "
                          f"print('Synthetic progress', file=sys.stderr); sys.exit({code})")
                if code:
                    with self.assertRaises(RuntimeError):
                        run_command([sys.executable, "-c", script], log_prefix=prefix)
                else:
                    output = run_command([sys.executable, "-c", script], log_prefix=prefix)
                    self.assertEqual(json.loads(output), {"value": 42})
                self.assertEqual(json.loads(Path(f"{prefix}.stdout.log").read_text()), {"value": 42})
                self.assertIn("Synthetic progress", Path(f"{prefix}.stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
