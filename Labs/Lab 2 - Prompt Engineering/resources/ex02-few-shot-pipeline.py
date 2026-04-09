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
        "role": "assistant",
        "content": '{"'
        'filing_type": "DEF 14A", '
        '"confidence": 0.95, "evidence": ["proxy statement", "advisory vote on executive compensation", "Named Executive Officers", "Annual Meeting of Shareholders"], '
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
            temperature=0.95
        )
        data = parse_json_response(response.choices[0].message.content)
        print(data)
        return ClassificationResult(**data)
    except:
        print("error")

def classify_few_shot(content: str) -> ClassificationResult:
    ...




if __name__ == "__main__":

    filings = load_filings()
    for f in filings:
        expected = f['filing_type']
        result = classify_zero_shot(f['content'])
        print(f"Expected: {expected}")
        if result.filing_type is not None:
            print(f"Result type: {result.filing_type}")
            print("=" * 40)
        else: 
            print("Result: None")