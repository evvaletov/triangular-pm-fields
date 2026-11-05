"""Extract the frozen numerical PM response fixture; no field computation.

python extract_pm_fixture.py --source references/arrays.npz --output pm_fixture.npz
Writes the NPZ and a sibling .manifest.json containing source and array hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED_SOURCE_SHA256 = '2be01e25ef4f505dd643d2b8dadcd88524de8984e8c56635d69d6c76cb44685e'


def extract(source, output):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError('Source NPZ does not match the frozen independently validated campaign')
    with np.load(source, allow_pickle=False) as archive:
        selected = {key: np.ascontiguousarray(archive[key][:64]) for key in (
            'nominal_holdout', 'geometry_sensitivity_coarse', 'geometry_sensitivity_fine',
            'holdout_points', 'tolerance_points')}
    np.testing.assert_array_equal(selected['holdout_points'], selected['tolerance_points'])
    np.savez_compressed(output, **selected)
    manifest = dict(source_npz_sha256=source_hash, output_npz_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    source_campaign='papers/triangular_pm_20261027/field_fit_20261103/references/arrays.npz',
                    selection='first64 entries along axis0 of each named source array',
                    interpretation='fixed numerical fixture for synthetic Gaussian verification; no new physical validation',
                    array_hash_convention='SHA256 of C-contiguous raw bytes; dtype and shape recorded separately',
                    arrays={key: dict(shape=list(value.shape), dtype=value.dtype.str,
                                      sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
                            for key, value in selected.items()})
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output), indent=2))
