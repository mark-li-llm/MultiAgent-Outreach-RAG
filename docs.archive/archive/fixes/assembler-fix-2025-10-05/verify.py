#!/usr/bin/env python3
"""
Verify the Assembler fix works correctly in run_graph.py context
"""
import re


def _word_count(t: str) -> int:
    """Count words in text"""
    return len(re.findall(r"\b\w+\b", t or ""))


def _grade(t: str) -> float:
    """Calculate Flesch-Kincaid readability grade"""
    sentences = [s for s in re.split(r"[.!?]+", t or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, _word_count(t))
    syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1
                           for w in re.findall(r"\b\w+\b", t or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59


def _shorten_body(b: str) -> str:
    """Shorten email body"""
    lines = b.splitlines()
    head = []
    bullets = []
    for ln in lines:
        if ln.strip().startswith("- "):
            bullets.append("- " + " ".join(ln.split()[1:9]))
        else:
            head.append(" ".join(ln.split()[:10]))
    bullets = bullets[:3]
    nb = "\n".join([ln for ln in head if ln.strip()] + bullets)
    return nb


def assembler_new_logic(body: str, compliance: dict) -> str:
    """
    New Assembler logic from run_graph.py (lines 739-783)
    Plan 1: Trust A2A with safeguards
    """
    current_wc = _word_count(body)
    current_grade = _grade(body)

    # Priority 1: Word count hard limit
    if current_wc > 160:
        iterations = 0
        while current_wc > 160 and iterations < 3:
            body = _shorten_body(body)
            current_wc = _word_count(body)
            iterations += 1

    # Priority 2: Readability grade with A2A trust
    elif current_grade > 15:  # Relaxed from 10 to 15
        # Check A2A result
        if compliance["flags"]["critical"] == []:
            # A2A passed - trust it, no truncation
            pass
        else:
            # A2A also flagged issues - try truncation with safeguard
            iterations = 0
            prev_grade = current_grade

            while current_grade > 10 and iterations < 3:
                new_body = _shorten_body(body)
                new_grade = _grade(new_body)

                # Safeguard: stop if grade gets worse
                if new_grade >= prev_grade:
                    break

                # Apply effective truncation
                body = new_body
                prev_grade = new_grade
                current_grade = new_grade
                iterations += 1

    # else: grade ≤ 15 and wc ≤ 160, no action needed

    return body


def main():
    print("=" * 70)
    print("Assembler Fix Verification (run_graph.py context)")
    print("=" * 70)

    # The actual email from test_fix2 A2A output
    original_email = """Here's how recent momentum could support your CX outcomes through data-driven, cross-channel service today, with scale.

- AI-enabled CRM on a unified platform connects data across sales, service, marketing, and IT to boost agent productivity, self-service, cross-channel CX.
- Agentforce 3 adds visibility and governance for AI agents with AgentExchange and templates, accelerating value and enabling more efficient workflows across teams.
- Regional deployment of Agentforce 3 (Canada, UK, India, Japan, Brazil) with GA governance supports consistent omnichannel CX and regional agent performance, reducing variability.

If you'd like, we can schedule a 15-minute call to discuss a pilot aligned to your CX targets."""

    # A2A compliance result (critical=[])
    compliance = {"flags": {"critical": [], "warning": []}}

    initial_wc = _word_count(original_email)
    initial_grade = _grade(original_email)

    print(f"\n📧 Original email:")
    print(f"   Word count: {initial_wc}")
    print(f"   Readability grade: {initial_grade:.2f}")
    print(f"   A2A flags: {compliance['flags']}")

    # Apply new Assembler logic
    result_email = assembler_new_logic(original_email, compliance)

    final_wc = _word_count(result_email)
    final_grade = _grade(result_email)

    print(f"\n✅ After Assembler (new logic):")
    print(f"   Word count: {final_wc}")
    print(f"   Readability grade: {final_grade:.2f}")
    print(f"   Email preserved: {result_email == original_email}")

    if result_email == original_email:
        print("\n🎉 SUCCESS! Email was preserved (trusted A2A)")
        print("\nExpected behavior:")
        print("  - Grade 18.73 > 15 → Triggers grade check")
        print("  - A2A passed (critical=[]) → Trust A2A, no truncation")
        print("  - Email remains intact at 108 words")
        return 0
    else:
        print("\n❌ FAIL! Email was modified")
        print(f"\nResult preview:")
        print(result_email[:200])
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
