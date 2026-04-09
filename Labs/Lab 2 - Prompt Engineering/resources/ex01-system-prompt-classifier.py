from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from starter import load_filings, parse_json_response, call_llm
import json

load_dotenv()


class FilingClassification(BaseModel):
    filing_type: str      # 10-K, 10-Q, 8-K, DEF 14A, S-1, UNKNOWN
    confidence: float     # 0.0 to 1.0
    evidence: list[str]   # Phrases from the text supporting the classification
    ambiguity_notes: str | None  # Explain uncertainty, or null



naive_system_prompt: str = "classify this filing accurately"
prod_system_prompt = '''
    You are an SEC filing classification system used by 
    financial analysts. Your sole task is to determine the filing type of a given 
    document excerpt and return a structured classification.
    Valid filing types (use these exact values):
    - 10-K: Annual report for a full fiscal year
    - 10-Q: Quarterly report for interim fiscal periods
    - 8-K: Current report for material events (officer changes, acquisitions, etc.)
    - DEF 14A: Proxy statement for shareholder meetings and executive compensation
    - S-1: Registration statement for initial public offerings
    - UNKNOWN: Use when the excerpt does not clearly match any of the above
    Respond with ONLY a JSON object matching this schema:
    {
        "filing_type": "10-K",        // one of: 10-K, 10-Q, 8-K, DEF 14A, S-1, UNKNOWN
        "confidence": 0.95,           // float between 0.0 and 1.0
        "evidence": ["phrase from text", ...],  // 1-3 phrases supporting your classification
        "ambiguity_notes": null       // string explaining uncertainty, or null if confident
    }
    Classification rules:
    - References to "quarterly period ended" or three-month financials → 10-Q
    - References to "fiscal year ended" or twelve-month financials → 10-K
    - Reports of sudden leadership changes, acquisitions, or bankruptcies → 8-K
    - Discussion of shareholder votes, proxy solicitation, or executive pay → DEF 14A
    - Language about "shares to be registered" or IPO prospectus → S-1
    If the excerpt is ambiguous or contains signals for multiple filing types,
    set filing_type to "UNKNOWN", confidence below 0.5, and explain the
    ambiguity in ambiguity_notes.
    Do not wrap the JSON in markdown code fences.
    Do not include any text before or after the JSON object.
    Do not add explanations, disclaimers, or commentary.
'''


if __name__ == "__main__":

    filings = load_filings()

    print("\n\nSYSTEM PROMPT COMPARISON: Naive vs. Production")
    print("=" * 45)

    # keys are [filename, company, sector, cik, filing_type, char_count, content]
    for f in filings:
        expected = f['filing_type']
        prod = call_llm(prod_system_prompt, f['content'])
        naive = call_llm(naive_system_prompt, f['content'])
        prod_data = parse_json_response(prod[0])
        naive_data = parse_json_response(naive[0])
        naive_type = naive_data.get("filing_type", "N/A") if naive_data else "N/A"
        naive_parseable = naive_data is not None
        prod_type = prod_data.get("filing_type", "N/A") if prod_data else "N/A"
        confidence = int(prod_data.get("confidence", 0) * 100) if prod_data else 0
        print(f'  {f['filename']} (expected: {expected})')
        print(f"    Naive:      {naive_type:<12} | JSON parseable: {'YES' if naive_parseable else 'NO'}")
        print(f"    Production: {prod_type:<12} | Confidence: {confidence}%")

    print("\n\nBATCH CLASSIFICATION")
    print("=" * 45)
    correct = 0
    total = len(filings)
    for f in filings:
        expected = f['filing_type']
        prod = call_llm(prod_system_prompt, f['content'])
        prod_data = parse_json_response(prod[0])

        if prod_data:
            result = FilingClassification(**prod_data)
            is_correct = result.filing_type == expected
            if is_correct:
                correct += 1
            print(
                f"  {f['filename']:<18} {result.filing_type:<10} "
                f"({int(result.confidence * 100)}%)  "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )
        else:
            print(f"  {f['filename']:<18} PARSE ERROR")

    print(f"\n  Accuracy: {correct}/{total} ({int(correct/total*100)}%)")