from starter import *
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv() 

class ClassificationResult(BaseModel):
    filing_type: str
    confidence: float
    evidence: list[str]
    ambiguity_notes: str | None

class ExtractedEntity(BaseModel):
    entity_type: str
    value: float
    context: str 

SYSTEM_PROMPT =  """
    You are an SEC filing classification system used by financial analysts.
    Your sole task is to determine the filing type of a given document excerpt.

    You MUST respond with ONLY a JSON object with exactly these fields:
    {
    "filing_type": "one of the exact values above",
    "confidence": float from 0 - 1,
    "evidence": ["phrase from text supporting classification"],
    "ambiguity_notes": notes or null
    }

    Do not wrap the JSON in markdown code fences.
    Do not include any text before or after the JSON object.
"""

FEW_SHOT_EXAMPLES = [
    # 10-K example
    {
        "role": "user",
        "content": "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934. "
        "For the fiscal year ended December 31, 2023. Total revenues for the twelve months ended "
        "December 31, 2023 were $4.2 billion, compared to $3.8 billion for the prior fiscal year. "
        "The company employed approximately 12,000 full-time employees as of year end."
    },
    {
        "role": "assistant",
        "content": '{'
        '"filing_type": "10-K", '
        '"confidence": 0.98, '
        '"evidence": ["fiscal year ended December 31, 2023", "twelve months ended", "ANNUAL REPORT PURSUANT TO SECTION 13"], '
        '"ambiguity_notes": null}'
    },

    # 10-Q example
    {
        "role": "user",
        "content": "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934. "
        "For the quarterly period ended September 30, 2023. Net income for the three months ended September 30, 2023 "
        "was $142 million, compared to $98 million for the same period last year. "
        "These interim financial statements do not include all disclosures required for annual financial statements."
    },
    {
        "role": "assistant",
        "content": '{'
        '"filing_type": "10-Q", '
        '"confidence": 0.97, '
        '"evidence": ["quarterly period ended September 30, 2023", "three months ended", "interim financial statements"], '
        '"ambiguity_notes": null}'
    },

    # 8-K example — boundary: could look like 10-K if you miss the trigger event
    {
        "role": "user",
        "content": "On March 15, 2024, the Board of Directors of Acme Corp accepted the resignation of John Smith "
        "as Chief Executive Officer, effective immediately. The Board has appointed Sarah Johnson, currently serving"
        " as Chief Operating Officer, as interim CEO while a search for a permanent replacement is conducted. "
        "This Current Report on Form 8-K is filed pursuant to Item 5.02."
    },
    {
        "role": "assistant",
        "content": '{'
        '"filing_type": "8-K", '
        '"confidence": 0.96, '
        '"evidence": ["Current Report on Form 8-K", "resignation of Chief Executive Officer", "Item 5.02"], '
        '"ambiguity_notes": null}'
    },

    # DEF 14A example — boundary: mentions compensation which also appears in 10-K
    {
        "role": "user",
        "content": "NOTICE OF ANNUAL MEETING OF SHAREHOLDERS. You are cordially invited to attend the Annual Meeting "
        "of Shareholders to vote on the election of directors and the advisory vote on executive compensation. "
        "The proxy statement contains information about the matters to be voted upon. The total compensation "
        "paid to our Named Executive Officers in fiscal 2023 is described in the Summary Compensation Table on page 42."
    },
    {
        "role": "assistant", "content": '{"'
        'filing_type": "DEF 14A", '
        '"confidence": 0.95, '
        '"evidence": ["proxy statement", "advisory vote on executive compensation", "Named Executive Officers", "Annual Meeting of Shareholders"], '
        '"ambiguity_notes": null}'
    },

    # S-1 example — boundary: mentions financials like a 10-K but context is IPO
    {
        "role": "user",
        "content": "We are offering 10,000,000 shares of our common stock in this initial public offering. Prior to this "
        "offering, there has been no public market for our common stock. We intend to apply to list our common stock on "
        "the Nasdaq Global Select Market under the symbol ACME. Investing in our common stock involves a high degree "
        "of risk. See Risk Factors beginning on page 15."
    },
    {
        "role": "assistant",
        "content": '{"'
        'filing_type": "S-1", '
        '"confidence": 0.97, '
        '"evidence": ["initial public offering", "no public market for our common stock", "shares to be registered"], '
        '"ambiguity_notes": null}'
    },

    # UNKNOWN example — boundary: contains signals for both 10-K and DEF 14A
    {
        "role": "user",
        "content": "The following financial data is derived from our audited consolidated financial statements. "
        "Executive compensation for fiscal year 2023 included base salary, annual bonus, and long-term equity awards."
        " Shareholders of record as of April 1, 2024 are entitled to vote. Total assets as of December 31, 2023"
        " were $2.1 billion representing a full year of operations."
    },
    {
        "role": "assistant",
        "content": '{"'
        'filing_type": "UNKNOWN", "'
        'confidence": 0.40, '
        '"evidence": ["fiscal year 2023", "shareholders entitled to vote", "executive compensation"], '
        '"ambiguity_notes": "Excerpt contains signals for both 10-K (full year financials, total assets) and DEF 14A '
        '(shareholder vote, executive compensation). Insufficient context to distinguish."}'
    },
]

agent = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)




def classify_zero_shot(content: str) -> ClassificationResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    try: 
        response = agent.chat.completions.create(
            model='gpt-4.1-nano',
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500
        )
        data = parse_json_response(response.choices[0].message.content)
        return ClassificationResult(**data)
    except:
        print("error")

def classify_few_shot(content: str) -> ClassificationResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": content}
    ]
    try: 
        response = agent.chat.completions.create(
            model='gpt-4.1-nano',
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens= 500
        )
        data = parse_json_response(response.choices[0].message.content)
        return ClassificationResult(**data)
    except:
        print("error")




if __name__ == "__main__":

    filings = load_filings()
    total = len(filings)
    zero_correct = 0
    few_correct = 0
    zero_total_input = 0
    zero_total_output = 0
    few_total_input = 0
    few_total_output = 0
    overhead_samples = []

    print("CLASSIFICATION: Zero-Shot vs. Few-Shot")
    print("=" * 40)
    for f in filings:

        few_output = 0
        expected = f['filing_type']
        filename = f['filename']

        z_content, z_input, z_output = call_llm(SYSTEM_PROMPT, f['content'])
        f_content, f_input, f_output = call_llm_few(SYSTEM_PROMPT, f['content'], extra_messages=FEW_SHOT_EXAMPLES)
        f_data = parse_json_response(f_content)
        z_data = parse_json_response(z_content)
        f_result = ClassificationResult(**f_data)
        z_result = ClassificationResult(**z_data)
        zero_type = z_result.filing_type if z_result else "ERROR"
        few_type = f_result.filing_type if f_result else "ERROR"

        zero_ok = zero_type == expected
        few_ok = few_type == expected

        # token tracking 
        zero_total_input += z_input
        zero_total_output += z_output
        few_total_input += f_input
        few_total_output += f_output
        overhead_samples.append(f_input - z_input)
        if zero_ok:
            zero_correct += 1
        if few_ok:
            few_correct += 1

        print(
            f"  {filename:<40} expected={expected:<10} "
            f"zero={zero_type:18} {'OK  ' if zero_ok else 'MISS'}| "
            f"few={few_type:<10} {'OK' if few_ok else 'MISS'}"
        )

    print()
    print(f"  Zero-shot accuracy: {zero_correct}/{total} ({int(zero_correct/total*100)}%)")
    print(f"  Few-shot accuracy:  {few_correct}/{total} ({int(few_correct/total*100)}%)")
    print("=" * 40)
    avg_overhead = int(sum(overhead_samples) / len(overhead_samples))
    zero_cost = estimate_cost(zero_total_input, zero_total_output)
    few_cost = estimate_cost(few_total_input, few_total_output)
    zero_cost_per = zero_cost / total
    few_cost_per = few_cost / total
    overhead_multiplier = few_cost / zero_cost if zero_cost > 0 else 0

    scale = 10_000
    zero_monthly = zero_cost_per * scale
    few_monthly = few_cost_per * scale
    diff_monthly = few_monthly - zero_monthly
    zero_misses = int((1 - zero_correct / total) * scale)
    few_misses = int((1 - few_correct / total) * scale)
    accuracy_gain = int((few_correct - zero_correct) / total * 100)
    cost_increase = int((overhead_multiplier - 1) * 100)

    print()
    print("FEW-SHOT COST ANALYSIS")
    print("======================")
    print(f"Per-filing overhead:   +{avg_overhead} input tokens (the few-shot examples)")
    print()
    print(f"                       {'Zero-Shot':<16} {'Few-Shot'}")
    print(f"Corpus input tokens:   {zero_total_input:<16,} {few_total_input:,}")
    print(f"Corpus output tokens:  {zero_total_output:<16,} {few_total_output:,}")
    print(f"Corpus total cost:     ${zero_cost:<15.4f} ${few_cost:.4f}")
    print(f"Cost per filing:       ${zero_cost_per:<15.6f} ${few_cost_per:.6f}")
    print(f"Overhead multiplier:   1.00x           {overhead_multiplier:.2f}x")
    print()
    print(f"PROJECTION @ {scale:,} filings/month:")
    print(f"  Zero-shot:  ${zero_monthly:.2f}/month")
    print(f"  Few-shot:   ${few_monthly:.2f}/month")
    print(f"  Difference: ${diff_monthly:.2f}/month")
    print()
    print("ACCURACY vs. COST TRADEOFF:")
    print(f"  Zero-shot accuracy: {zero_correct}/{total} ({int(zero_correct/total*100)}%)  →  ~{zero_misses:,} misclassifications/month")
    print(f"  Few-shot accuracy:  {few_correct}/{total} ({int(few_correct/total*100)}%)  →  ~{few_misses:,} misclassifications/month")
    print(f"  Accuracy gain:      +{accuracy_gain}%  for  +{cost_increase}% cost")