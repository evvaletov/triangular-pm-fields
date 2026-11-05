"""Content-addressed persistent TAN_CACHE_V1 geometry cache.

``load_or_prepare(triangles, charges, directory, *, coalesce=True)`` returns an
immutable ``CacheArtifact`` with path, sha256, metadata, and hit/rebuild
disposition. A verified hit reads and hashes the saved TAN text and does not
construct Fraction geometry.

Identity binds canonical C-contiguous little-endian binary64 input bytes and
shapes, charges, coalesce, TAN_CACHE_V1, store schema, and the SHA-256 of the
``prepare_cache`` producer loaded into this process. Invalid shapes and
nonfinite values are rejected before hashing. The loaded producer is bound at
import; later disk edits are source drift and are rejected until reload. The
loaded module content must match the co-located ``prepare_cache.py``.

Entries are JSON plus text, never pickle. Publication is atomic per file with a
per-entry interprocess lock. Malformed, stale, or corrupt records rebuild with
an explicit reason. This addresses accidental corruption and staleness, not
authenticated storage against malicious rewrites. There is no numerical field
cache here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import fcntl
import hashlib
import json
import os
import tempfile

import numpy as np

from . import prepare_cache
from .prepare_cache import PreparedCache


FORMAT_VERSION = 'TAN_CACHE_V1'
STORE_SCHEMA = 1
CONSTRUCTOR_NAME = 'prepare_cache.py'
GEOMETRY_NAME = 'geometry.tan'
MANIFEST_NAME = 'manifest.json'
FLOAT64_LE = np.dtype('<f8')
REQUIRED_MANIFEST = (
    'format', 'store_schema', 'coalesce', 'source_sha256', 'input_identity',
    'triangles_shape', 'charges_shape', 'triangles_sha256', 'charges_sha256',
    'content_sha256', 'prepared_metadata',
)


class _CountMismatch(ValueError):
    """Well-typed TAN header or metadata counts disagree with the payload."""


def _loaded_producer_sha256():
    fingerprint = getattr(prepare_cache, '_source_fingerprint', None)
    if callable(fingerprint):
        return fingerprint()
    path = Path(prepare_cache.__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


_LOADED_PRODUCER_SHA256 = _loaded_producer_sha256()
_COLOCATED_PRODUCER = Path(__file__).resolve().parent / CONSTRUCTOR_NAME


@dataclass(frozen=True)
class CacheArtifact:
    """Immutable on-disk TAN geometry record. Not a PreparedCache."""
    path: Path
    sha256: str
    metadata: MappingProxyType
    disposition: str
    reason: str


def constructor_source_sha256(path=None):
    """Fingerprint of the loaded producer, or of an explicit source path.

    The default is the producer imported into this process, bound at import.
    It is not a live hash of the on-disk file. An explicit ``path`` hashes that
    file without claiming it is loaded. Source drift and co-location mismatch
    raise ``ValueError`` for the default path.
    """
    if path is not None:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    _assert_producer_identity()
    return _LOADED_PRODUCER_SHA256


def _assert_producer_identity():
    loaded_path = Path(prepare_cache.__file__).resolve()
    try:
        loaded_bytes = loaded_path.read_bytes()
    except OSError as exc:
        raise ValueError('loaded prepare_cache source is unreadable') from exc
    loaded_digest = hashlib.sha256(loaded_bytes).hexdigest()
    if loaded_digest != _LOADED_PRODUCER_SHA256:
        raise ValueError(
            'loaded prepare_cache source drifted since import; '
            'reload the module or start a new process')
    try:
        colocated_bytes = _COLOCATED_PRODUCER.read_bytes()
    except OSError as exc:
        raise ValueError('co-located prepare_cache.py is unreadable') from exc
    if colocated_bytes != loaded_bytes:
        raise ValueError(
            'loaded prepare_cache does not match co-located producer; '
            'sys.path shadowing cannot label cache identity')


def _f64_le_bytes(array):
    """Canonical little-endian IEEE-754 binary64 bytes, C-contiguous."""
    return np.ascontiguousarray(array, dtype=FLOAT64_LE).tobytes()


def canonicalize_inputs(triangles, charges):
    """Copy to finite C-contiguous binary64 arrays; reject bad shape/nonfinite."""
    tri = np.array(triangles, dtype=np.float64, copy=True)
    sigma = np.array(charges, dtype=np.float64, copy=True)
    if tri.size == 0:
        tri = np.empty((0, 3, 3), dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or sigma.shape != (len(tri),):
        raise ValueError('expected triangles(m,3,3), charges(m)')
    if not np.isfinite(tri).all() or not np.isfinite(sigma).all():
        raise ValueError('nonfinite input')
    return np.ascontiguousarray(tri), np.ascontiguousarray(sigma)


def input_identity(triangles, charges, *, coalesce, source_sha256):
    tri, sigma = canonicalize_inputs(triangles, charges)
    return _identity_from_canonical(tri, sigma, bool(coalesce), source_sha256)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _identity_from_canonical(tri, sigma, coalesce, source_sha256):
    payload = [
        FORMAT_VERSION.encode('ascii'),
        f'store_schema={STORE_SCHEMA}'.encode('ascii'),
        f'coalesce={1 if coalesce else 0}'.encode('ascii'),
        f'source={source_sha256}'.encode('ascii'),
        f'triangles_shape={",".join(map(str, tri.shape))}'.encode('ascii'),
        f'charges_shape={",".join(map(str, sigma.shape))}'.encode('ascii'),
        b'dtype=<f8',
        _f64_le_bytes(tri),
        _f64_le_bytes(sigma),
    ]
    return hashlib.sha256(b'\n'.join(payload)).hexdigest()


def _artifact(path, sha256, prepared, extra, disposition, reason):
    metadata = dict(prepared)
    metadata.update(extra)
    metadata['disposition'] = disposition
    metadata['reason'] = reason
    metadata['sha256'] = sha256
    return CacheArtifact(path=Path(path), sha256=sha256,
                         metadata=MappingProxyType(metadata),
                         disposition=disposition, reason=reason)


class _EntryLock:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _as_int_list(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(name)
    out = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(name)
        out.append(item)
    return out


def _as_sha(value, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(name)
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(name) from exc
    if value != value.lower():
        raise ValueError(name)
    return value


def _nonneg_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name)
    return value


def _prepared_metadata(value):
    if not isinstance(value, dict):
        raise ValueError('prepared_metadata')
    required = ('input_triangles', 'planes', 'boundary_edges', 'sha256',
                'exact_setup', 'query_geometry_precision')
    for key in required:
        if key not in value:
            raise ValueError('prepared_metadata')
    _nonneg_int(value['input_triangles'], 'prepared_metadata.input_triangles')
    _nonneg_int(value['planes'], 'prepared_metadata.planes')
    _nonneg_int(value['boundary_edges'], 'prepared_metadata.boundary_edges')
    _as_sha(value['sha256'], 'prepared_metadata.sha256')
    if value['exact_setup'] is not True:
        raise ValueError('prepared_metadata')
    if not isinstance(value['query_geometry_precision'], str):
        raise ValueError('prepared_metadata')
    return {key: value[key] for key in required}


def _verify_tan_header_counts(text_bytes, planes, edges):
    """Check TAN magic, np/ne header, and record count without Fractions."""
    try:
        text = text_bytes.decode('utf-8')
    except UnicodeError as exc:
        raise ValueError('tan_header') from exc
    if not text.startswith(FORMAT_VERSION + '\n') or not text.endswith('\n'):
        raise ValueError('tan_header')
    lines = text.split('\n')
    parts = lines[1].split(' ')
    if len(parts) != 2:
        raise ValueError('tan_header')
    try:
        header_planes = int(parts[0], 10)
        header_edges = int(parts[1], 10)
    except ValueError as exc:
        raise ValueError('tan_header') from exc
    if str(header_planes) != parts[0] or str(header_edges) != parts[1]:
        raise ValueError('tan_header')
    if header_planes < 0 or header_edges < 0:
        raise ValueError('tan_header')
    nrec = len(lines) - 3
    if nrec < 0:
        raise ValueError('tan_header')
    if header_planes != planes or header_edges != edges or nrec != planes + edges:
        raise _CountMismatch('tan header or record count inconsistent with metadata')


def _load_manifest(path):
    raw = path.read_text(encoding='utf-8')
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('manifest is not an object')
    missing = [key for key in REQUIRED_MANIFEST if key not in data]
    if missing:
        raise ValueError('missing ' + ','.join(missing))
    return data, raw


def _verify_entry(entry, *, identity, source, coalesce, tri, sigma):
    manifest_path = entry / MANIFEST_NAME
    geometry_path = entry / GEOMETRY_NAME
    has_manifest = manifest_path.is_file()
    has_geometry = geometry_path.is_file()
    if not has_manifest and not has_geometry:
        return None, 'absent'
    if not has_manifest:
        return None, 'missing_manifest'
    if not has_geometry:
        return None, 'missing_geometry'
    try:
        data, _raw = _load_manifest(manifest_path)
        coalesce_value = data['coalesce']
        if coalesce_value is not True and coalesce_value is not False:
            return None, 'malformed_manifest'
        if data['format'] != FORMAT_VERSION:
            return None, 'stale_format'
        schema = data['store_schema']
        if isinstance(schema, bool) or not isinstance(schema, int):
            return None, 'malformed_manifest'
        if schema != STORE_SCHEMA:
            return None, 'stale_schema'
        if coalesce_value is not coalesce:
            return None, 'option_mismatch'
        if _as_sha(data['source_sha256'], 'source_sha256') != source:
            return None, 'source_mismatch'
        if _as_sha(data['input_identity'], 'input_identity') != identity:
            return None, 'identity_mismatch'
        if _as_int_list(data['triangles_shape'], 3, 'triangles_shape') != list(tri.shape):
            return None, 'shape_mismatch'
        if _as_int_list(data['charges_shape'], 1, 'charges_shape') != list(sigma.shape):
            return None, 'shape_mismatch'
        if _as_sha(data['triangles_sha256'], 'triangles_sha256') != _sha256_bytes(_f64_le_bytes(tri)):
            return None, 'input_mismatch'
        if _as_sha(data['charges_sha256'], 'charges_sha256') != _sha256_bytes(_f64_le_bytes(sigma)):
            return None, 'input_mismatch'
        prepared = _prepared_metadata(data['prepared_metadata'])
        if prepared['input_triangles'] != int(sigma.shape[0]):
            return None, 'count_mismatch'
        text = geometry_path.read_bytes()
        digest = _sha256_bytes(text)
        content = _as_sha(data['content_sha256'], 'content_sha256')
        if digest != content:
            return None, 'content_hash_mismatch'
        if prepared['sha256'] != digest:
            return None, 'content_hash_mismatch'
        try:
            _verify_tan_header_counts(text, prepared['planes'], prepared['boundary_edges'])
        except _CountMismatch:
            return None, 'count_mismatch'
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None, 'malformed_manifest'
    extra = dict(
        format=FORMAT_VERSION,
        store_schema=STORE_SCHEMA,
        coalesce=coalesce,
        source_sha256=source,
        cache_key=identity,
        input_triangles=prepared['input_triangles'],
        planes=prepared['planes'],
        boundary_edges=prepared['boundary_edges'],
        exact_setup=prepared['exact_setup'],
        query_geometry_precision=prepared['query_geometry_precision'],
    )
    return (geometry_path, digest, prepared, extra, text), 'verified'


def _publish(entry, text, prepared, identity, source, coalesce, tri, sigma):
    digest = _sha256_bytes(text.encode('utf-8'))
    if prepared['sha256'] != digest:
        raise ValueError('constructor sha256 does not match serialized text')
    manifest = dict(
        format=FORMAT_VERSION,
        store_schema=STORE_SCHEMA,
        coalesce=bool(coalesce),
        source_sha256=source,
        input_identity=identity,
        triangles_shape=[int(x) for x in tri.shape],
        charges_shape=[int(x) for x in sigma.shape],
        triangles_sha256=_sha256_bytes(_f64_le_bytes(tri)),
        charges_sha256=_sha256_bytes(_f64_le_bytes(sigma)),
        content_sha256=digest,
        constructor=CONSTRUCTOR_NAME,
        prepared_metadata=dict(prepared),
    )
    entry.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(entry / GEOMETRY_NAME, text)
    _atomic_write_text(entry / MANIFEST_NAME, json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    _fsync_dir(entry)
    return digest


def load_or_prepare(triangles, charges, directory, *, coalesce=True):
    """Load a verified TAN cache or prepare and publish one.

    Returns a ``CacheArtifact``. Disposition is ``hit`` when the saved text
    verifies, ``miss`` when no entry existed, and ``rebuild`` when a stale or
    corrupt entry was replaced. A hit never constructs ``PreparedCache``.
    """
    if coalesce is not True and coalesce is not False:
        raise ValueError('coalesce must be a bool')
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError('cache directory must be a directory')
    tri, sigma = canonicalize_inputs(triangles, charges)
    _assert_producer_identity()
    source = _LOADED_PRODUCER_SHA256
    identity = _identity_from_canonical(tri, sigma, coalesce, source)
    directory.mkdir(parents=True, exist_ok=True)
    _fsync_dir(directory)
    entry = directory / identity
    lock = directory / '.locks' / identity
    extra_base = dict(
        format=FORMAT_VERSION,
        store_schema=STORE_SCHEMA,
        coalesce=coalesce,
        source_sha256=source,
        cache_key=identity,
    )
    with _EntryLock(lock):
        verified, reason = _verify_entry(
            entry, identity=identity, source=source, coalesce=coalesce,
            tri=tri, sigma=sigma)
        if verified is not None:
            path, digest, prepared, extra, _text = verified
            return _artifact(path, digest, prepared, extra, 'hit', reason)
        cache = PreparedCache(tri, sigma, coalesce=coalesce)
        digest = _publish(entry, cache.text, cache.metadata(), identity, source,
                          coalesce, tri, sigma)
        disposition = 'miss' if reason == 'absent' else 'rebuild'
        extra = dict(extra_base)
        extra.update(input_triangles=len(cache.triangles), planes=len(cache.planes),
                     boundary_edges=len(cache.edges),
                     exact_setup=True,
                     query_geometry_precision=cache.metadata()['query_geometry_precision'])
        return _artifact(entry / GEOMETRY_NAME, digest, cache.metadata(), extra,
                         disposition, 'absent' if disposition == 'miss' else reason)
