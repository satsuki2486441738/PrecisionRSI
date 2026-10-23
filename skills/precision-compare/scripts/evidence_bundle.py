#!/usr/bin/env python3
"""Pack, verify and safely extract indexed evidence ZIPs with the standard library.
Call: pack(source_dir, bundle_path); verify(bundle_path); extract(bundle_path, output)
Input: Selected files or an indexed evidence ZIP; optional literal text replacements.
Output: Portable bundle or verified/extracted receipts with hashes and sizes.
Requires: Standard library; new destinations, portable relative paths; default expanded-byte cap 16 GiB.

Details:
API: pack(source_dir, bundle_path, replacements=None, *, max_expanded_bytes=...),
verify(bundle_path, index='EXPORT.json', *, max_expanded_bytes=...), and
extract(bundle_path, output, index='EXPORT.json', *, max_expanded_bytes=...).
All return JSON-serializable receipts. Outputs must be new paths; extraction
verifies every member before creating its output. Invalid input raises ValueError,
OSError or zipfile.BadZipFile. No bundled code or serialized models are loaded.
Windows filesystem operations use extended-length absolute paths internally;
receipts retain readable absolute paths and archive names remain relative POSIX.

pack accepts an ordered mapping or sequence of (literal, replacement) string
pairs. Replacement is sequential, UTF-8-only, across all filename extensions
except .npy/.npz/.safetensors/.pt/.pth/.bin/.gguf/.f16/.bf16/.f32/.f64.
Invalid UTF-8 and NUL-containing files
are copied exactly. Processing is streamed, with transformed text spooled to disk
as needed. Patterns and replacement values are never stored in EXPORT.json;
each file records original/exported SHA256 and byte counts plus a replacement
count. Paths themselves are preserved. This is not privacy certification:
selection, private filenames and scanning beyond explicit literals are the
caller's responsibility. There are no automatic address or other regex rules.

EXPORT.json: {schema_version:1, files:[{path, original_sha256, exported_sha256,
original_bytes, exported_bytes, literal_replacement_count}, ...]}. Its files list
excludes itself and must cover all other members exactly. Verification also
accepts a selected plain index such as FILES.json: {path:{sha256,bytes}, ...}.
Legacy EXPORT files-list entries without exported_bytes/bytes use ZIP metadata;
receipts disclose their count. Provided hashes/sizes are always validated.

Only regular-file members with unique portable relative POSIX paths are allowed;
directories, symlinks, traversal and case/Unicode-normalization collisions fail.
The default expanded-byte cap is 16 GiB, including the index; change it through
max_expanded_bytes or --max-expanded-bytes. Index JSON is additionally limited
to 16 MiB. Caps apply both to ZIP metadata and streamed reads. CLI subcommands:
pack SOURCE NEW.zip [--replacements PAIRS.json], verify BUNDLE [--index FILES.json],
extract BUNDLE NEW_DIRECTORY [--index FILES.json]. PAIRS.json is an object or
ordered pair list. Receipts go to stdout; exit 0=success, 1=invalid input/error.
"""
import argparse
import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile


DEFAULT_MAX_EXPANDED_BYTES = 16 * 1024**3
MAX_INDEX_BYTES = 16 * 1024**2
CHUNK_BYTES = 1024 * 1024
_BINARY_SUFFIXES = {'.npy', '.npz', '.safetensors', '.pt', '.pth', '.bin', '.gguf',
                    '.f16', '.bf16', '.f32', '.f64'}
_RESERVED = {'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'} | {
    prefix + number for prefix in ('com', 'lpt') for number in '123456789\u00b9\u00b2\u00b3'}
_SHA256 = re.compile(r'[0-9a-fA-F]{64}\Z')


def _readable_path(path):
    value = os.fspath(path)
    if os.name == 'nt':
        if value.startswith('\\\\?\\UNC\\'):
            value = '\\\\' + value[8:]
        elif value.startswith('\\\\?\\'):
            value = value[4:]
    return Path(os.path.abspath(value))


def _filesystem_path(path):
    """Normalize first: extended Windows paths do not resolve dot components."""
    absolute = _readable_path(path)
    if os.name != 'nt':
        return absolute
    value = str(absolute)
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def _cap(value):
    if type(value) is not int or value <= 0:
        raise ValueError('max_expanded_bytes must be a positive integer')
    return value


def _safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid portable member path')
    parts = name.split('/')
    for part in parts:
        if (not part or part in ('.', '..') or part[-1] in ' .'
                or any(ord(c) < 32 or ord(c) == 127 or c in '<>:"|?*' for c in part)
                or part.split('.')[0].rstrip(' ').casefold() in _RESERVED):
            raise ValueError('invalid portable member path: ' + repr(name))
    return name


def _name_key(name):
    return unicodedata.normalize('NFC', name).casefold()


def _validate_names(names):
    seen = {}
    for name in names:
        _safe_name(name)
        key = _name_key(name)
        if key in seen:
            raise ValueError('duplicate or colliding member paths: ' + repr(name))
        seen[key] = name
    for key, name in seen.items():
        parts = key.split('/')
        if any('/'.join(parts[:i]) in seen for i in range(1, len(parts))):
            raise ValueError('file/directory member path collision: ' + repr(name))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key: ' + repr(key))
        result[key] = value
    return result


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite JSON value')
    return result


def _parse_json(data):
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_float=_finite_float,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError('nonfinite JSON value')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('index must be valid UTF-8 JSON') from exc


def _digest(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError('invalid SHA256 for ' + label)
    return value.lower()


def _size(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('invalid byte count for ' + label)
    return value


def _stream(source, target=None, *, limit):
    digest = hashlib.sha256()
    size = 0
    while True:
        data = source.read(min(CHUNK_BYTES, limit - size + 1))
        if not data:
            break
        size += len(data)
        if size > limit:
            raise ValueError('expanded-byte cap exceeded')
        digest.update(data)
        if target is not None:
            target.write(data)
    return digest.hexdigest(), size


def _archive_digest(stream):
    position = stream.tell()
    stream.seek(0)
    digest, _ = _stream(stream, limit=2**63 - 1)
    stream.seek(position)
    return digest


def _archive_members(archive, index, cap):
    _safe_name(index)
    infos = archive.infolist()
    _validate_names([info.orig_filename for info in infos])
    total = 0
    for info in infos:
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (info.filename != info.orig_filename or info.is_dir()
                or info.external_attr & 0x10 or kind not in (0, stat.S_IFREG)):
            raise ValueError('nonregular archive member: ' + repr(info.orig_filename))
        if info.flag_bits & 1:
            raise ValueError('encrypted archive member: ' + repr(info.filename))
        total += _size(info.file_size, info.filename)
        if total > cap:
            raise ValueError('expanded-byte cap exceeded')
    members = {info.filename: info for info in infos}
    if index not in members:
        raise ValueError('missing archive index: ' + index)
    if members[index].file_size > MAX_INDEX_BYTES:
        raise ValueError('index exceeds 16 MiB limit')
    return members


def _index_records(value, index, members):
    records = {}
    missing_sizes = 0
    if not isinstance(value, dict):
        raise ValueError('index must be a JSON object')
    if isinstance(value.get('files'), list):
        entries = value['files']
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('index files must contain objects')
            name = _safe_name(entry.get('path'))
            if name in records:
                raise ValueError('duplicate index path: ' + name)
            digest = _digest(entry.get('exported_sha256'), name)
            size = entry.get('exported_bytes', entry.get('bytes'))
            if size is None:
                missing_sizes += 1
            else:
                _size(size, name)
            if 'original_sha256' in entry:
                _digest(entry['original_sha256'], name)
            if 'original_bytes' in entry:
                _size(entry['original_bytes'], name)
            records[name] = (digest, size)
    else:
        for name, entry in value.items():
            _safe_name(name)
            if not isinstance(entry, dict):
                raise ValueError('plain index records must be objects')
            records[name] = (_digest(entry.get('sha256'), name),
                             _size(entry.get('bytes'), name))
    _validate_names(list(records))
    if set(records) != set(members) - {index}:
        raise ValueError('archive index coverage mismatch')
    for name, (digest, size) in list(records.items()):
        if size is not None and size != members[name].file_size:
            raise ValueError('archive byte count mismatch: ' + name)
        records[name] = (digest, members[name].file_size if size is None else size)
    return records, missing_sizes


def _verify_open(stream, index, cap):
    archive = zipfile.ZipFile(stream)
    try:
        members = _archive_members(archive, index, cap)
        with archive.open(index) as source:
            data = source.read(min(cap, MAX_INDEX_BYTES) + 1)
        if len(data) != members[index].file_size or len(data) > min(cap, MAX_INDEX_BYTES):
            raise ValueError('index byte count or cap mismatch')
        records, missing_sizes = _index_records(_parse_json(data), index, members)
        expanded = len(data)
        for name, (expected_hash, expected_size) in records.items():
            with archive.open(name) as source:
                digest, size = _stream(source, limit=cap - expanded)
            expanded += size
            if size != expected_size or digest != expected_hash:
                raise ValueError('archive hash or byte count mismatch: ' + name)
        return archive, members, records, data, missing_sizes
    except BaseException:
        archive.close()
        raise


def _receipt(action, bundle, index, records, digest, missing_sizes=0):
    return dict(action=action, bundle=str(_readable_path(bundle)), index=index,
                file_count=len(records), payload_bytes=sum(size for _, size in records.values()),
                archive_sha256=digest, entries_without_indexed_sizes=missing_sizes,
                size_source='zip_metadata' if missing_sizes else 'index')


def verify(bundle_path, index='EXPORT.json', *, max_expanded_bytes=DEFAULT_MAX_EXPANDED_BYTES):
    """Check safe paths, exact index coverage and every streamed hash/size."""
    cap = _cap(max_expanded_bytes)
    with _filesystem_path(bundle_path).open('rb') as stream:
        archive, _, records, _, missing_sizes = _verify_open(stream, index, cap)
        try:
            return _receipt('verify', bundle_path, index, records,
                            _archive_digest(stream), missing_sizes)
        finally:
            archive.close()


def extract(bundle_path, output, index='EXPORT.json', *, max_expanded_bytes=DEFAULT_MAX_EXPANDED_BYTES):
    """Verify first, then extract to a new directory; remove it on write failure."""
    cap = _cap(max_expanded_bytes)
    output = _filesystem_path(output)
    if os.path.lexists(output):
        raise FileExistsError('extraction output already exists: ' + str(_readable_path(output)))
    with _filesystem_path(bundle_path).open('rb') as stream:
        archive, members, records, index_data, missing_sizes = _verify_open(stream, index, cap)
        try:
            digest = _archive_digest(stream)
            output.mkdir(parents=True, exist_ok=False)
            try:
                expected = dict(records)
                expected[index] = (hashlib.sha256(index_data).hexdigest(), len(index_data))
                expanded = 0
                for name in members:
                    target = output.joinpath(*name.split('/'))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open('xb') as destination:
                        actual_hash, size = _stream(source, destination, limit=cap - expanded)
                    expanded += size
                    if (actual_hash, size) != expected[name]:
                        raise ValueError('archive changed during extraction: ' + name)
            except BaseException:
                shutil.rmtree(output)
                raise
            receipt = _receipt('extract', bundle_path, index, records, digest, missing_sizes)
            receipt['output'] = str(_readable_path(output))
            return receipt
        finally:
            archive.close()


class _LiteralStage:
    def __init__(self, pattern, replacement):
        self.pattern, self.replacement = pattern, replacement
        self.pending = ''
        self.count = 0

    def feed(self, text, final=False):
        text = self.pending + text
        pieces = []
        start = 0
        while True:
            found = text.find(self.pattern, start)
            if found < 0:
                break
            pieces.extend((text[start:found], self.replacement))
            self.count += 1
            start = found + len(self.pattern)
        end = len(text) if final else max(start, len(text) - len(self.pattern) + 1)
        pieces.append(text[start:end])
        self.pending = text[end:]
        return ''.join(pieces)


def _replacement_pairs(replacements):
    if replacements is None:
        return []
    values = replacements.items() if isinstance(replacements, dict) else replacements
    result = []
    try:
        for pair in values:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('replacements must be literal/replacement string pairs')
            pattern, replacement = pair
            if not isinstance(pattern, str) or not pattern or not isinstance(replacement, str):
                raise ValueError('replacement patterns must be nonempty strings')
            pattern.encode('utf-8')
            replacement.encode('utf-8')
            result.append((pattern, replacement))
    except TypeError as exc:
        raise ValueError('replacements must be an ordered mapping or pair sequence') from exc
    return result


def _transformed(path, pairs, limit):
    """Return a seekable export spool, provenance, and count; binary returns None."""
    output = tempfile.SpooledTemporaryFile(max_size=CHUNK_BYTES)
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    stages = [_LiteralStage(*pair) for pair in pairs]
    original_hash, exported_hash = hashlib.sha256(), hashlib.sha256()
    original_size = exported_size = 0
    try:
        with path.open('rb') as source:
            while True:
                data = source.read(CHUNK_BYTES)
                original_hash.update(data)
                original_size += len(data)
                if b'\0' in data:
                    output.close()
                    return None
                text = decoder.decode(data, final=not data)
                for stage in stages:
                    text = stage.feed(text, final=not data)
                encoded = text.encode('utf-8')
                exported_size += len(encoded)
                if exported_size > limit:
                    raise ValueError('expanded-byte cap exceeded')
                output.write(encoded)
                exported_hash.update(encoded)
                if not data:
                    break
    except UnicodeError:
        output.close()
        return None
    except BaseException:
        output.close()
        raise
    output.seek(0)
    return output, original_hash.hexdigest(), original_size, exported_hash.hexdigest(), exported_size, sum(
        stage.count for stage in stages)


def _source_files(source):
    source_info = source.lstat()
    if (not stat.S_ISDIR(source_info.st_mode)
            or getattr(source_info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
        raise ValueError('source must be a regular directory')
    files = []
    for directory, directories, filenames in os.walk(source, followlinks=False):
        for name in directories + filenames:
            path = Path(directory) / name
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
                raise ValueError('source links/reparse points are not supported')
            if name in filenames:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('source contains a nonregular file')
                files.append((path.relative_to(source).as_posix(), path))
    _validate_names([name for name, _ in files] + ['EXPORT.json'])
    return sorted(files)


def pack(source_dir, bundle_path, replacements=None, *, max_expanded_bytes=DEFAULT_MAX_EXPANDED_BYTES):
    """Create a new ZIP and provenance index, applying only explicit text literals."""
    cap = _cap(max_expanded_bytes)
    pairs = _replacement_pairs(replacements)
    source, bundle = _filesystem_path(source_dir), _filesystem_path(bundle_path)
    if bundle.resolve().is_relative_to(source.resolve()):
        raise ValueError('bundle must be outside its source directory')
    files = _source_files(source)
    entries, records = [], {}
    expanded = 0
    with bundle.open('xb') as stream:
        try:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for name, path in files:
                    transformed = _transformed(path, pairs, cap - expanded) if pairs and path.suffix.casefold() not in _BINARY_SUFFIXES else None
                    if transformed is None:
                        with path.open('rb') as original, archive.open(name, 'w', force_zip64=True) as destination:
                            digest, size = _stream(original, destination, limit=cap - expanded)
                        original_hash, original_size, count = digest, size, 0
                    else:
                        exported, original_hash, original_size, digest, size, count = transformed
                        try:
                            with archive.open(name, 'w', force_zip64=True) as destination:
                                copied_hash, copied_size = _stream(exported, destination, limit=cap - expanded)
                            if (copied_hash, copied_size) != (digest, size):
                                raise ValueError('export spool changed')
                        finally:
                            exported.close()
                    expanded += size
                    records[name] = (digest, size)
                    entries.append(dict(path=name, original_sha256=original_hash, exported_sha256=digest,
                                        original_bytes=original_size, exported_bytes=size,
                                        literal_replacement_count=count))
                data = (json.dumps(dict(schema_version=1, files=entries), ensure_ascii=False,
                                   indent=2, allow_nan=False) + '\n').encode('utf-8')
                if len(data) > MAX_INDEX_BYTES or expanded + len(data) > cap:
                    raise ValueError('index size or expanded-byte cap exceeded')
                archive.writestr('EXPORT.json', data)
        except BaseException:
            stream.close()
            bundle.unlink()
            raise
    with bundle.open('rb') as stream:
        return _receipt('pack', bundle, 'EXPORT.json', records, _archive_digest(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('pack', 'verify', 'extract'):
        command = commands.add_parser(action)
        command.add_argument('source' if action == 'pack' else 'bundle', type=Path)
        if action == 'pack':
            command.add_argument('bundle', type=Path)
            command.add_argument('--replacements', type=Path)
        elif action == 'extract':
            command.add_argument('output', type=Path)
        if action != 'pack':
            command.add_argument('--index', default='EXPORT.json')
        command.add_argument('--max-expanded-bytes', type=int, default=DEFAULT_MAX_EXPANDED_BYTES)
    args = parser.parse_args(argv)
    try:
        options = dict(max_expanded_bytes=args.max_expanded_bytes)
        if args.action == 'pack':
            replacements = _parse_json(_filesystem_path(args.replacements).read_bytes()) if args.replacements else None
            receipt = pack(args.source, args.bundle, replacements, **options)
        elif args.action == 'verify':
            receipt = verify(args.bundle, args.index, **options)
        else:
            receipt = extract(args.bundle, args.output, args.index, **options)
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
