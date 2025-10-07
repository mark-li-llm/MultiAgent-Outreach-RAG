#!/usr/bin/env python3
"""
Verify that non-SEC chunks were not modified during XBRL removal.
Compares hashes of backup chunks vs current chunks.
"""

import hashlib
import json
from pathlib import Path

def compute_file_hash(file_path):
    """Compute SHA256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    # Find the latest backup directory
    backup_dirs = sorted(Path('data/backup').glob('chunks_*'))
    if not backup_dirs:
        print("ERROR: No backup directory found")
        return 1

    backup_dir = backup_dirs[-1]
    current_dir = Path('data/interim/chunks')

    print(f"Comparing chunks:")
    print(f"  Backup: {backup_dir}")
    print(f"  Current: {current_dir}")
    print()

    # Get all non-SEC chunk files from backup
    sec_patterns = ['10-K', '10-Q', '8-K']
    backup_files = [f for f in backup_dir.glob('*.chunks.jsonl')
                    if not any(pattern in f.name for pattern in sec_patterns)]

    print(f"Found {len(backup_files)} non-SEC chunk files in backup")

    # Compare hashes
    unchanged = []
    changed = []
    missing = []

    for backup_file in backup_files:
        current_file = current_dir / backup_file.name

        if not current_file.exists():
            missing.append(backup_file.name)
            continue

        backup_hash = compute_file_hash(backup_file)
        current_hash = compute_file_hash(current_file)

        if backup_hash == current_hash:
            unchanged.append(backup_file.name)
        else:
            changed.append(backup_file.name)

    # Report results
    print(f"\nResults:")
    print(f"  ✓ Unchanged: {len(unchanged)} files")
    print(f"  ✗ Changed: {len(changed)} files")
    print(f"  ✗ Missing: {len(missing)} files")

    if changed:
        print(f"\nChanged files:")
        for f in changed[:10]:  # Show first 10
            print(f"  - {f}")
        if len(changed) > 10:
            print(f"  ... and {len(changed) - 10} more")

    if missing:
        print(f"\nMissing files:")
        for f in missing[:10]:
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Summary
    print("\n" + "=" * 80)
    if changed or missing:
        print("RESULT: ✗ FAIL - Non-SEC chunks were modified or missing")
        return 1
    else:
        print("RESULT: ✓ PASS - All non-SEC chunks unchanged")
        return 0

if __name__ == "__main__":
    exit(main())
