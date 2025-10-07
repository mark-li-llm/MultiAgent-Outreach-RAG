#!/usr/bin/env python3
"""
Update ground truth after re-chunking (2025-10-07).
Re-chunking reduced chunks from 565 → 536.
All 10-K/10-Q ground truth chunk IDs need updating.
"""
import json
from pathlib import Path

# Updated mappings after re-chunking
RECHUNK_FIXES = {
    # 10-Q queries (FY26 Q1) - 79 chunks (was 91)
    "10q_q1_revenue": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0015",
    "10q_operating_cash": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0036",
    "10q_senior_notes": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0036",  # Same chunk as cash flow
    "10q_share_repurchase": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0020",
    "10q_ai_risks": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0059",
    "10q_data_privacy": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0061",

    # 10-K queries (FY25) - 116 chunks (was 133)
    "10k_fy25_performance": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0058",
    "10k_agentforce_strategy": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0002",
    "10k_sales_cloud_offerings": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0004",
}

def main():
    eval_seed_path = Path("data/interim/eval/salesforce_eval_seed.jsonl")

    # Backup
    backup_path = eval_seed_path.with_suffix(".jsonl.backup_rechunking_20251007")
    if not backup_path.exists():
        import shutil
        shutil.copy(eval_seed_path, backup_path)
        print(f"✓ Backed up to: {backup_path}")

    # Read and update
    updated_lines = []
    changes = 0

    with open(eval_seed_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            eval_id = entry.get("eval_id")

            if eval_id in RECHUNK_FIXES:
                old_chunk_id = entry["expected_chunk_id"]
                new_chunk_id = RECHUNK_FIXES[eval_id]

                if old_chunk_id != new_chunk_id:
                    entry["expected_chunk_id"] = new_chunk_id

                    # Update notes
                    old_note = entry.get("notes", "")
                    entry["notes"] = f"{old_note} [GT re-corrected after re-chunking 2025-10-07: {old_chunk_id.split('::')[-1]} -> {new_chunk_id.split('::')[-1]}]".strip()

                    changes += 1
                    print(f"✓ {eval_id}: {old_chunk_id.split('::')[-1]} → {new_chunk_id.split('::')[-1]}")

            updated_lines.append(json.dumps(entry, ensure_ascii=False))

    # Write updated file
    with open(eval_seed_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines) + '\n')

    print(f"\n✓ Updated {changes} ground truth annotations")
    print(f"✓ Backup: {backup_path}")
    print(f"✓ Updated: {eval_seed_path}")

if __name__ == "__main__":
    main()
