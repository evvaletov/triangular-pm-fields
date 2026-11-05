"""Persistent geometry-cache regressions. No numerical field cache is tested."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import triangular_pm_fields.cache_store as cache_store
from triangular_pm_fields.cache_store import CacheArtifact, load_or_prepare
from triangular_pm_fields.prepare_cache import PreparedCache


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

RECTANGLE = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
                      [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], dtype=np.float64)


def rectangle_mesh(n, *, bounds=(-1.0, 1.0, -0.75, 0.75), z=0.0, sigma=0.8):
    xs = np.linspace(bounds[0], bounds[1], n + 1, dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], n + 1, dtype=np.float64)
    triangles = []
    for i in range(n):
        for j in range(n):
            a = np.array([xs[i], ys[j], z], dtype=np.float64)
            b = np.array([xs[i + 1], ys[j], z], dtype=np.float64)
            c = np.array([xs[i + 1], ys[j + 1], z], dtype=np.float64)
            d = np.array([xs[i], ys[j + 1], z], dtype=np.float64)
            triangles.extend(([a, b, c], [a, c, d]))
    tri = np.array(triangles, dtype=np.float64)
    return tri, np.full(len(tri), sigma, dtype=np.float64)


def _mp_prepare(payload):
    directory, triangles, charges, coalesce = payload
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from triangular_pm_fields.cache_store import load_or_prepare as child_load
    artifact = child_load(triangles, charges, directory, coalesce=coalesce)
    return (artifact.sha256, artifact.path.read_bytes(), artifact.disposition,
            artifact.reason, str(artifact.path), dict(artifact.metadata))


class PersistentCacheStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / 'tan_cache'

    def test_miss_then_hit_same_bytes(self):
        miss = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(miss.disposition, 'miss')
        self.assertEqual(miss.reason, 'absent')
        self.assertIsInstance(miss, CacheArtifact)
        self.assertTrue(miss.path.is_file())
        expected = PreparedCache(RECTANGLE, [1, 1])
        self.assertEqual(miss.path.read_bytes(), expected.text.encode())
        self.assertEqual(miss.sha256, expected.sha256)
        self.assertEqual(miss.metadata['planes'], 1)
        self.assertEqual(miss.metadata['boundary_edges'], 4)
        hit = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(hit.disposition, 'hit')
        self.assertEqual(hit.reason, 'verified')
        self.assertEqual(hit.sha256, miss.sha256)
        self.assertEqual(hit.path.read_bytes(), miss.path.read_bytes())
        self.assertEqual(hit.path, miss.path)
        self.assertEqual(hit.metadata['planes'], miss.metadata['planes'])
        self.assertEqual(hit.metadata['cache_key'], miss.metadata['cache_key'])

    def test_hit_does_not_invoke_prepared_cache(self):
        first = load_or_prepare(RECTANGLE, [1.0, 1.0], self.directory)
        self.assertEqual(first.disposition, 'miss')

        def explode(*args, **kwargs):
            raise AssertionError('PreparedCache invoked on a verified hit')

        with mock.patch.object(cache_store, 'PreparedCache', side_effect=explode):
            hit = load_or_prepare(RECTANGLE, [1.0, 1.0], self.directory)
        self.assertEqual(hit.disposition, 'hit')
        self.assertEqual(hit.sha256, first.sha256)
        self.assertEqual(hit.path.read_bytes(), first.path.read_bytes())

    def test_bit_shape_charge_coalesce_and_source_mismatch(self):
        base = load_or_prepare(RECTANGLE, [1, 1], self.directory, coalesce=True)
        shifted = RECTANGLE.copy()
        shifted[0, 0, 0] = np.nextafter(shifted[0, 0, 0], 1.0)
        bit = load_or_prepare(shifted, [1, 1], self.directory, coalesce=True)
        self.assertEqual(bit.disposition, 'miss')
        self.assertNotEqual(bit.metadata['cache_key'], base.metadata['cache_key'])
        self.assertNotEqual(bit.path, base.path)

        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE.reshape(6, 3), [1, 1], self.directory)

        charge = load_or_prepare(RECTANGLE, [1, -1], self.directory, coalesce=True)
        self.assertEqual(charge.disposition, 'miss')
        self.assertNotEqual(charge.sha256, base.sha256)

        option = load_or_prepare(RECTANGLE, [1, 1], self.directory, coalesce=False)
        self.assertEqual(option.disposition, 'miss')
        self.assertNotEqual(option.metadata['cache_key'], base.metadata['cache_key'])
        # n=1 has no collinear joins, so TAN bytes may match; n=2 changes them.
        fine, fine_c = rectangle_mesh(2)
        with_join = load_or_prepare(fine, fine_c, self.directory, coalesce=True)
        without = load_or_prepare(fine, fine_c, self.directory, coalesce=False)
        self.assertNotEqual(with_join.metadata['cache_key'], without.metadata['cache_key'])
        self.assertNotEqual(with_join.sha256, without.sha256)
        self.assertNotEqual(with_join.metadata['boundary_edges'],
                            without.metadata['boundary_edges'])

        manifest = json.loads((base.path.parent / 'manifest.json').read_text())
        manifest['source_sha256'] = '0' * 64
        (base.path.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory, coalesce=True)
        self.assertEqual(rebuilt.disposition, 'rebuild')
        self.assertEqual(rebuilt.reason, 'source_mismatch')
        self.assertEqual(rebuilt.sha256, base.sha256)

    def test_altered_and_truncated_text_and_json(self):
        artifact = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        original = artifact.path.read_bytes()

        artifact.path.write_bytes(original[:-5])
        truncated = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(truncated.disposition, 'rebuild')
        self.assertEqual(truncated.reason, 'content_hash_mismatch')
        self.assertEqual(truncated.path.read_bytes(), original)

        artifact.path.write_bytes(original.replace(b'1', b'2', 1))
        altered = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(altered.disposition, 'rebuild')
        self.assertEqual(altered.reason, 'content_hash_mismatch')

        manifest_path = artifact.path.parent / 'manifest.json'
        manifest_path.write_text('{')
        bad_json = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(bad_json.disposition, 'rebuild')
        self.assertEqual(bad_json.reason, 'malformed_manifest')

        payload = json.loads(manifest_path.read_text())
        payload['content_sha256'] = 'a' * 64
        manifest_path.write_text(json.dumps(payload, indent=2) + '\n')
        bad_hash = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(bad_hash.disposition, 'rebuild')
        self.assertEqual(bad_hash.reason, 'content_hash_mismatch')

        payload = json.loads(manifest_path.read_text())
        payload['input_identity'] = 'b' * 64
        manifest_path.write_text(json.dumps(payload, indent=2) + '\n')
        aliased = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(aliased.disposition, 'rebuild')
        self.assertEqual(aliased.reason, 'identity_mismatch')

    def test_input_mutation_does_not_alias(self):
        triangles = RECTANGLE.copy()
        first = load_or_prepare(triangles, [1, 1], self.directory)
        triangles[0, 1, 1] = 0.5
        mutated = load_or_prepare(triangles, [1, 1], self.directory)
        self.assertEqual(mutated.disposition, 'miss')
        self.assertNotEqual(mutated.metadata['cache_key'], first.metadata['cache_key'])
        restored = RECTANGLE.copy()
        hit = load_or_prepare(restored, [1, 1], self.directory)
        self.assertEqual(hit.disposition, 'hit')
        self.assertEqual(hit.metadata['cache_key'], first.metadata['cache_key'])

    def test_same_and_different_keys(self):
        a = load_or_prepare(RECTANGLE, [1, 1], self.directory, coalesce=True)
        b = load_or_prepare(np.array(RECTANGLE, dtype=np.float64), np.array([1.0, 1.0]),
                            self.directory, coalesce=True)
        self.assertEqual(a.metadata['cache_key'], b.metadata['cache_key'])
        self.assertEqual(b.disposition, 'hit')
        other_dir = Path(self.tmp.name) / 'other'
        c = load_or_prepare(RECTANGLE, [1, 1], other_dir, coalesce=True)
        self.assertEqual(c.disposition, 'miss')
        self.assertEqual(c.sha256, a.sha256)
        self.assertNotEqual(c.path, a.path)

    def test_two_writer_processes_same_artifact(self):
        tri, charges = rectangle_mesh(2)
        payload = (str(self.directory), tri.tolist(), charges.tolist(), True)
        ctx = mp.get_context('spawn')
        with ctx.Pool(2) as pool:
            rows = pool.map(_mp_prepare, [payload, payload])
        sha, texts, dispositions, reasons, paths, metas = zip(*rows)
        self.assertEqual(set(sha), {sha[0]})
        self.assertEqual(texts[0], texts[1])
        self.assertEqual(set(paths), {paths[0]})
        self.assertTrue(set(dispositions) <= {'hit', 'miss'})
        self.assertIn('miss', dispositions)
        expected = PreparedCache(tri, charges).text.encode()
        self.assertEqual(texts[0], expected)
        verified = load_or_prepare(tri, charges, self.directory)
        self.assertEqual(verified.disposition, 'hit')
        self.assertEqual(verified.path.read_bytes(), expected)

    def test_interruption_partial_entry_rebuilds(self):
        first = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        key = first.metadata['cache_key']
        entry = self.directory / key
        original = first.path.read_bytes()

        first.path.unlink()
        (entry / 'geometry.tan.tmp.interrupted').write_bytes(original[:20])
        missing_geometry = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(missing_geometry.disposition, 'rebuild')
        self.assertEqual(missing_geometry.reason, 'missing_geometry')
        self.assertEqual(missing_geometry.path.read_bytes(), original)

        (entry / 'manifest.json').unlink()
        first.path.write_bytes(original[:12])
        missing_manifest = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(missing_manifest.disposition, 'rebuild')
        self.assertEqual(missing_manifest.reason, 'missing_manifest')

        shutil.rmtree(entry)
        leftover = self.directory / (key + '.partial')
        leftover.mkdir()
        (leftover / 'geometry.tan').write_bytes(original[:8])
        absent = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual(absent.disposition, 'miss')
        self.assertEqual(absent.path.read_bytes(), original)

    def test_rejects_invalid_before_hashing(self):
        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE, [1], self.directory)
        self.assertFalse(self.directory.exists())
        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE * np.nan, [1, 1], self.directory)
        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE, [np.inf, 1], self.directory)
        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE, [1, 1], self.directory, coalesce='yes')
        file_path = Path(self.tmp.name) / 'not_a_dir'
        file_path.write_text('x')
        with self.assertRaises(ValueError):
            load_or_prepare(RECTANGLE, [1, 1], file_path)

    def test_empty_and_canceled_inputs(self):
        empty = load_or_prepare(np.empty((0, 3, 3)), np.empty((0,)), self.directory)
        self.assertEqual(empty.disposition, 'miss')
        self.assertEqual(empty.path.read_text(), 'TAN_CACHE_V1\n0 0\n')
        hit = load_or_prepare([], [], self.directory)
        self.assertEqual(hit.disposition, 'hit')
        doubled = np.concatenate([RECTANGLE, RECTANGLE])
        canceled = load_or_prepare(doubled, [1, 1, -1, -1], self.directory)
        self.assertEqual(canceled.metadata['planes'], 0)
        self.assertEqual(canceled.metadata['boundary_edges'], 0)

    def test_loaded_source_drift_rejects_until_new_process(self):
        src = Path(self.tmp.name) / 'isolated_src'
        src.mkdir()
        pkg = src / 'triangular_pm_fields'
        pkg.mkdir()
        (pkg / '__init__.py').write_text('')
        here = Path(cache_store.__file__).resolve().parent
        shutil.copy(here / 'prepare_cache.py', pkg / 'prepare_cache.py')
        shutil.copy(here / 'cache_store.py', pkg / 'cache_store.py')
        cache_dir = Path(self.tmp.name) / 'isolated_cache'
        triangles = json.dumps(RECTANGLE.tolist())
        charges = json.dumps([1.0, 1.0])
        script = (
            'import json,sys\n'
            'from pathlib import Path\n'
            'import numpy as np\n'
            'from triangular_pm_fields.cache_store import load_or_prepare, constructor_source_sha256\n'
            'tri=np.array(json.loads(sys.argv[1]), dtype=np.float64)\n'
            'sigma=np.array(json.loads(sys.argv[2]), dtype=np.float64)\n'
            'directory=sys.argv[3]\n'
            'mode=sys.argv[4]\n'
            'out={"source": constructor_source_sha256()}\n'
            'art=load_or_prepare(tri, sigma, directory)\n'
            'out["key"]=art.metadata["cache_key"]\n'
            'out["sha"]=art.sha256\n'
            'out["disposition"]=art.disposition\n'
            'if mode=="edit_then_reuse":\n'
            '    p=Path("triangular_pm_fields/prepare_cache.py")\n'
            '    p.write_bytes(p.read_bytes()+b"\\n# source-drift marker\\n")\n'
            '    try:\n'
            '        load_or_prepare(tri, sigma, directory)\n'
            '        out["after_edit"]="accepted"\n'
            '    except ValueError as exc:\n'
            '        out["after_edit"]="rejected"\n'
            '        out["after_edit_message"]=str(exc)\n'
            'print(json.dumps(out))\n'
        )
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src)
        first = subprocess.run(
            [sys.executable, '-c', script, triangles, charges, str(cache_dir),
             'edit_then_reuse'],
            cwd=str(src), env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        first_out = json.loads(first.stdout)
        self.assertEqual(first_out['disposition'], 'miss')
        self.assertEqual(first_out['after_edit'], 'rejected')
        self.assertIn('drifted since import', first_out['after_edit_message'])
        second = subprocess.run(
            [sys.executable, '-c', script, triangles, charges, str(cache_dir),
             'fresh'],
            cwd=str(src), env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        second_out = json.loads(second.stdout)
        self.assertEqual(second_out['disposition'], 'miss')
        self.assertNotEqual(second_out['source'], first_out['source'])
        self.assertNotEqual(second_out['key'], first_out['key'])
        self.assertEqual(second_out['sha'], first_out['sha'])

        shadow = Path(self.tmp.name) / 'shadow_prepare'
        shadow.mkdir()
        shutil.copy(pkg / 'prepare_cache.py', shadow / 'prepare_cache.py')
        (shadow / 'prepare_cache.py').write_bytes(
            (shadow / 'prepare_cache.py').read_bytes() + b'\n# shadow\n')
        # Restore co-located producer to the first-copy content.
        shutil.copy(here / 'prepare_cache.py', pkg / 'prepare_cache.py')
        # Deliberately inject an alternate package search path; ordinary top-level
        # sys.path shadowing is already excluded by the relative import.
        (pkg / '__init__.py').write_text('__path__.insert(0, ' + repr(str(shadow)) + ')\n')
        shadow_script = (
            'from triangular_pm_fields.cache_store import load_or_prepare\n'
            'import json,sys,numpy as np\n'
            'tri=np.array(json.loads(sys.argv[1]), dtype=np.float64)\n'
            'sigma=np.array(json.loads(sys.argv[2]), dtype=np.float64)\n'
            'try:\n'
            '    load_or_prepare(tri, sigma, sys.argv[3])\n'
            '    print(json.dumps({"status":"accepted"}))\n'
            'except ValueError as exc:\n'
            '    print(json.dumps({"status":"rejected","message":str(exc)}))\n'
        )
        shadow_env = os.environ.copy()
        shadow_env['PYTHONPATH'] = str(shadow) + os.pathsep + str(src)
        shadowed = subprocess.run(
            [sys.executable, '-c', shadow_script, triangles, charges,
             str(Path(self.tmp.name) / 'shadow_cache')],
            cwd=str(Path(self.tmp.name)), env=shadow_env, capture_output=True,
            text=True, timeout=60)
        self.assertEqual(shadowed.returncode, 0, shadowed.stdout + shadowed.stderr)
        shadow_out = json.loads(shadowed.stdout)
        self.assertEqual(shadow_out['status'], 'rejected')
        self.assertIn('co-located producer', shadow_out['message'])

    def test_manifest_counts_schema_and_header_rebuild(self):
        artifact = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        original = artifact.path.read_bytes()
        manifest_path = artifact.path.parent / 'manifest.json'

        def payload():
            return json.loads(manifest_path.read_text())

        def write(data):
            manifest_path.write_text(json.dumps(data, indent=2) + '\n')

        data = payload()
        data['prepared_metadata']['planes'] = -1
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'malformed_manifest'))
        self.assertEqual(rebuilt.path.read_bytes(), original)

        data = payload()
        data['prepared_metadata']['boundary_edges'] = -4
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'malformed_manifest'))

        data = payload()
        data['prepared_metadata']['input_triangles'] = -2
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'malformed_manifest'))

        data = payload()
        data['prepared_metadata']['input_triangles'] = 99
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'count_mismatch'))

        data = payload()
        data['store_schema'] = True
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'malformed_manifest'))

        data = payload()
        data['prepared_metadata']['planes'] = data['prepared_metadata']['planes'] + 1
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'count_mismatch'))

        data = payload()
        lines = original.decode('utf-8').split('\n')
        lines[1] = '2 4'
        altered = '\n'.join(lines)
        artifact.path.write_text(altered)
        digest = hashlib.sha256(altered.encode('utf-8')).hexdigest()
        data['content_sha256'] = digest
        data['prepared_metadata']['sha256'] = digest
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'count_mismatch'))
        self.assertEqual(rebuilt.path.read_bytes(), original)

        data = payload()
        extra = original.decode('utf-8') + '1 0 0 0 0 0 0 0\n'
        artifact.path.write_text(extra)
        digest = hashlib.sha256(extra.encode('utf-8')).hexdigest()
        data['content_sha256'] = digest
        data['prepared_metadata']['sha256'] = digest
        write(data)
        rebuilt = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertEqual((rebuilt.disposition, rebuilt.reason),
                         ('rebuild', 'count_mismatch'))

    def test_artifact_is_not_prepared_cache(self):
        artifact = load_or_prepare(RECTANGLE, [1, 1], self.directory)
        self.assertFalse(hasattr(artifact, 'planes'))
        self.assertFalse(hasattr(artifact, 'edges'))
        self.assertFalse(hasattr(artifact, 'text'))
        with self.assertRaises(AttributeError):
            artifact.sha256 = 'x'
        with self.assertRaises(TypeError):
            artifact.metadata['disposition'] = 'mutated'


if __name__ == '__main__':
    unittest.main(verbosity=2)
