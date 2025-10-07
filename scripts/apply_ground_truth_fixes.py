#!/usr/bin/env python3
"""
Apply ground truth fixes to eval seed file.
"""
import json
from typing import Dict

# Mapping of eval_id to corrected chunk_id
FIXES = {
    # 10-Q queries (FY26 Q1)
    "10q_q1_revenue": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0008",
    "10q_operating_cash": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0009",
    "10q_senior_notes": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0049",
    "10q_share_repurchase": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0046",
    "10q_ai_risks": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0070",
    "10q_data_privacy": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0073",

    # 10-K queries (FY25)
    "10k_fy25_performance": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0079",
    "10k_agentforce_strategy": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0014",
    "10k_sales_cloud_offerings": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0013",
}

def main():
    input_file = "data/interim/eval/salesforce_eval_seed.jsonl"
    output_file = "data/interim/eval/salesforce_eval_seed_corrected.jsonl"

    queries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries")

    fixed_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            eval_id = query.get('eval_id')

            if eval_id in FIXES:
                old_chunk = query.get('expected_chunk_id')
                new_chunk = FIXES[eval_id]
                query['expected_chunk_id'] = new_chunk
                query['notes'] = f"{query.get('notes', '')} [GT corrected: {old_chunk} -> {new_chunk}]"
                fixed_count += 1
                print(f"Fixed {eval_id}: {old_chunk} -> {new_chunk}")

            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    print(f"\nFixed {fixed_count} queries")
    print(f"Corrected seed saved to: {output_file}")

if __name__ == '__main__':
    main()
