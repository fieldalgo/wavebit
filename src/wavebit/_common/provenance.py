"""Source and environment identities for local research records."""
import hashlib
from importlib import metadata
import io
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_identity(files):
    identity = hashlib.sha256()
    for name, row in sorted(files.items()):
        identity.update(name.encode('utf-8') + b'\0' + row['sha256'].encode('ascii') + b'\n')
    return identity.hexdigest()


def environment():
    packages = {}
    for name in ('wavebit', 'numpy', 'scipy', 'array-api-compat', 'jax', 'jaxlib', 'torch'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'implementation': sys.implementation.name,
            'system': platform.system(), 'machine': platform.machine(), 'packages': packages}


def _git(root, *arguments):
    try:
        process = subprocess.run(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false',
                                  '-C', str(root), *arguments], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout if process.returncode == 0 else None


def capture_source(package_root, *, include_snapshot=True):
    """Hash installed Python source and runtime data; never inspect unrelated files."""
    package_root = Path(package_root).resolve()
    paths = set(package_root.rglob('*.py')) | set(package_root.rglob('*.pyi'))
    data_root = package_root / 'data'
    if data_root.is_dir():
        paths.update(p for p in data_root.rglob('*') if p.is_file())
    payloads = {}
    for path in sorted(paths):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(package_root):
            raise ValueError(f'Source snapshots do not follow symbolic links: {path.name}')
        name = 'wavebit/' + path.relative_to(package_root).as_posix()
        payloads[name] = path.read_bytes()
    files = {name: {'sha256': sha256(data), 'bytes': len(data)} for name, data in payloads.items()}
    repo_raw = _git(package_root, 'rev-parse', '--show-toplevel')
    git_info, patch = None, None
    if repo_raw:
        repo = Path(repo_raw.decode().strip())
        # A site-packages directory can itself be under an unrelated checkout.
        # Only claim Git provenance when the installed package has tracked files.
        relative = package_root.relative_to(repo).as_posix() if package_root.is_relative_to(repo) else None
        tracked = _git(repo, 'ls-files', '-z', '--', relative) if relative else None
        if tracked:
            head = _git(repo, 'rev-parse', '--verify', 'HEAD')
            status = _git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all', '--', relative)
            # The patch has the same code/data scope as the ZIP, including
            # tracked deletions. Never copy arbitrary files from the package.
            eligible = []
            for name in tracked.decode().split('\0'):
                if not name:
                    continue
                member = Path(name).relative_to(relative)
                if member.suffix in ('.py', '.pyi') or member.parts[0] == 'data':
                    eligible.append(name)
            patch = (_git(repo, 'diff', '--no-ext-diff', '--no-textconv', '--binary', 'HEAD', '--', *eligible)
                     if eligible else None)
            git_info = {'commit': head.decode().strip() if head else None,
                        'package_path': relative, 'package_dirty': bool(status) if status is not None else None,
                        'patch_scope': 'tracked package code/data relative to HEAD; untracked source is in source.zip'}
    snapshot = None
    if include_snapshot:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in payloads.items():
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        snapshot = stream.getvalue()
    info = {'identity': source_identity(files), 'files': files, 'git': git_info,
            'scope': 'installed wavebit Python source, type stubs and runtime data',
            'snapshot_included': include_snapshot,
            'observation': 'files on disk at capture time; not an attestation of previously imported bytecode'}
    return info, snapshot, patch
