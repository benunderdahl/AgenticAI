from pydantic import BaseModel
from dotenv import load_dotenv
from starter import *
import time
from pathlib import Path

load_dotenv()

class ComplianceIssue(BaseModel):
    issue: str
    severity: str       # HIGH, MEDIUM, LOW
    regulation: str     # Specific regulation or section
    evidence: str       # Quote from the filing

class ComplianceAssessment(BaseModel):
    issues: list[ComplianceIssue]
    overall_risk: str   # HIGH, MEDIUM, LOW
    reasoning_steps: list[str] | None = None # Only present in CoT

DIRECT_PROMPT = """
You are a compliance reviewer for SEC filings.
Review the provided 10-K filing excerpt for regulatory compliance issues.

Return ONLY a JSON object with this structure:
{
  "issues": [
    {
      "issue": "description of the issue",
      "severity": "HIGH, MEDIUM, or LOW",
      "regulation": "specific regulation or section",
      "evidence": "quote or reference from the filing"
    }
  ],
  "overall_risk": "HIGH, MEDIUM, or LOW",
  "reasoning_steps": null
}

Do not wrap in markdown. Return only the JSON object.
"""

COT_PROMPT = """
You are an expert SEC compliance reviewer.
Analyze the provided 10-K filing by working through each step below before rendering judgment.

STEP 1 - MD&A Analysis (Reg S-K Item 303):
  Check for: revenue/margin trend discussion, forward-looking trajectory,
  liquidity discussion, known trends and uncertainties.

STEP 2 - Risk Factor Analysis (Reg S-K Item 105):
  Check for: specificity of risk factors, boilerplate language,
  missing emerging risks (climate, AI/ML, cybersecurity),
  proportionality to company's actual risk profile.

STEP 3 - SOX Certification Review (Sections 302 and 906):
  Check for: proper CEO/CFO certification language,
  tenure of certifying officers, interim officer concerns,
  internal controls assessment completeness.

STEP 4 - Filing Timeliness Review:
  Check for: officer changes requiring 8-K within 4 business days,
  material events that should have triggered 8-K filings,
  consistency of dates mentioned.

STEP 5 - Cross-Reference Consistency:
  Check for: internal references that are inconsistent or circular,
  sections referenced that don't contain expected content,
  numbering or labeling inconsistencies.

After completing all 5 steps, return ONLY a JSON object:
{
  "issues": [
    {
      "issue": "description of the issue",
      "severity": "HIGH, MEDIUM, or LOW",
      "regulation": "specific regulation or section",
      "evidence": "quote or reference from the filing"
    }
  ],
  "overall_risk": "HIGH, MEDIUM, or LOW",
  "reasoning_steps": [
    "Step 1 finding: ...",
    "Step 2 finding: ...",
    "Step 3 finding: ...",
    "Step 4 finding: ...",
    "Step 5 finding: ..."
  ]
}

Do not wrap in markdown. Return only the JSON object.
"""

def direct_review(document: str) -> tuple[ComplianceAssessment | None, int, int, float]:
    start = time.time()
    content, input_tokens, output_tokens = call_llm(
        DIRECT_PROMPT,
        document,
        max_tokens=1000
    )
    elapsed = time.time() - start
    data = parse_json_response(content)
    if data:
        return ComplianceAssessment(**data), input_tokens, output_tokens, elapsed
    return None, input_tokens, output_tokens, elapsed


def cot_review(document: str) -> tuple[ComplianceAssessment | None, int, int, float]:
    start = time.time()
    content, input_tokens, output_tokens = call_llm(
        COT_PROMPT,
        document,
        max_tokens=2000  # CoT needs more output tokens
    )
    elapsed = time.time() - start
    data = parse_json_response(content)
    if data:
        return ComplianceAssessment(**data), input_tokens, output_tokens, elapsed
    return None, input_tokens, output_tokens, elapsed


if __name__ == "__main__":
    COMPLIANCE_DOCUMENT = (Path(__file__).parent / "compliance-scenario.txt").read_text(encoding="utf-8")

    print("Running direct review...")
    direct_result, d_in, d_out, d_time = direct_review(COMPLIANCE_DOCUMENT)

    print("Running chain-of-thought review...")
    cot_result, c_in, c_out, c_time = cot_review(COMPLIANCE_DOCUMENT)

    d_cost = estimate_cost(d_in, d_out)
    c_cost = estimate_cost(c_in, c_out)
    cost_multiplier = c_cost / d_cost if d_cost > 0 else 0

    d_issues = direct_result.issues if direct_result else []
    c_issues = cot_result.issues if cot_result else []

    print()
    print("COMPLIANCE REVIEW COMPARISON")
    print("=" * 40)
    print()
    print(f"DIRECT: {len(d_issues)} issues found | ${d_cost:.6f} | {d_time:.1f}s")
    print(f"COT:    {len(c_issues)} issues found | ${c_cost:.6f} | {c_time:.1f}s")
    print(f"Cost multiplier: {cost_multiplier:.1f}x")
    print()

    # issues only found by CoT
    direct_issue_texts = {i.issue.lower()[:40] for i in d_issues}
    cot_only = [i for i in c_issues if i.issue.lower()[:40] not in direct_issue_texts]

    if cot_only:
        print()
        print("Issues found only by CoT:")
        for i in cot_only:
            print(f"  - [{i.severity}] {i.issue}")

    # token breakdown
    print()
    print("TOKEN BREAKDOWN")
    print("-" * 40)
    print(f"{'':20} {'Direct':>10} {'CoT':>10}")
    print(f"{'Input tokens':20} {d_in:>10,} {c_in:>10,}")
    print(f"{'Output tokens':20} {d_out:>10,} {c_out:>10,}")
    print(f"{'Total tokens':20} {d_in+d_out:>10,} {c_in+c_out:>10,}")
    print(f"{'Cost (USD)':20} ${d_cost:>9.6f} ${c_cost:>9.6f}")

    # CoT reasoning steps
    if cot_result and cot_result.reasoning_steps:
        print()
        print("COT REASONING STEPS:")
        print("-" * 40)
        for step in cot_result.reasoning_steps:
            print(f"  {step}")

    # all issues
    print()
    print("ALL ISSUES FOUND")
    print("-" * 40)
    print("DIRECT:")
    for i in d_issues:
        print(f"  [{i.severity}] {i.issue}")
        print(f"         Reg: {i.regulation}")
        print(f"         Evidence: {i.evidence[:80]}")
        print()

    print("COT:")
    for i in c_issues:
        print(f"  [{i.severity}] {i.issue}")
        print(f"         Reg: {i.regulation}")
        print(f"         Evidence: {i.evidence[:80]}")
        print()