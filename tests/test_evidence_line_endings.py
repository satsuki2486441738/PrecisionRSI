"""Frozen evidence must survive a Git checkout with Windows line conversion."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class EvidenceLineEndingsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('git'), 'Git is required for checkout verification')
    def test_autocrlf_preserves_frozen_bytes_but_converts_ordinary_text(self):
        attributes = Path(__file__).resolve().parents[1] / '.gitattributes'
        with tempfile.TemporaryDirectory(prefix='precision-evidence-eol-') as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            self.assertTrue(root.name.startswith('precision-evidence-eol-'))
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git('init', '--quiet')
            git('config', 'core.autocrlf', 'true')
            git('config', 'core.eol', 'crlf')
            (root / '.gitattributes').write_bytes(attributes.read_bytes())
            evidence = Path('site/evidence/fixture/selected-targets.json')
            (root / evidence).parent.mkdir(parents=True)
            frozen = b'{\n  "target_logprobs": [-1.0, -2.0]\n}\n'
            (root / evidence).write_bytes(frozen)
            ordinary = Path('ordinary.txt')
            (root / ordinary).write_bytes(b'first\nsecond\n')
            git('add', '--', '.gitattributes', evidence.as_posix(), ordinary.as_posix())
            output = root / 'checkout'
            git('checkout-index', '--prefix=' + output.as_posix() + '/', '--',
                evidence.as_posix(), ordinary.as_posix())
            checked = (output / evidence).read_bytes()
            self.assertEqual(checked, frozen)
            self.assertEqual(hashlib.sha256(checked).digest(), hashlib.sha256(frozen).digest())
            self.assertEqual((output / ordinary).read_bytes(), b'first\r\nsecond\r\n')


if __name__ == '__main__':
    unittest.main()
