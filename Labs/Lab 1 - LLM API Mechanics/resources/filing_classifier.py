from pydantic import BaseModel, ValidationError
import json
from openai import OpenAI, AzureOpenAI
from starter import load_all_filings
import tiktoken as tt
import os
from dotenv import load_dotenv


PRICING = {
    "gpt-4o":             {"input": 2.50, "output": 10.00},   # USD per 1M tokens
    "claude-3.5-sonnet":  {"input": 3.00, "output": 15.00},
}

MODEL_LIMITS = {
    "gpt-4o":            {"context_window": 128_000},
    "claude-3.5-sonnet": {"context_window": 200_000},
}


class FilingClassification(BaseModel):
    filing_type: str    # Expected values: 10-K, 10-Q, 8-K, DEF 14A
    confidence: float   # 0.0 to 1.0
    company_name: str
    primary_sector: str
    reasoning: str      # One sentence explaining the classification
   

   
def classify_filing(text: str, temperature: float, client: AzureOpenAI) -> FilingClassification:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify SEC filings and return ONLY JSON with:\n"
                "filing_type (str), confidence (float between 0.0 and 1.0, NOT words like 'high'), "
                "company_name (str), primary_sector (str), reasoning (str, one sentence)"
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=200
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return FilingClassification(**data)

    except (ValidationError, json.JSONDecodeError) as e:
        print(e)


def experiment_runner():
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    filings = load_all_filings()

    # sample 5 unique filing types
    sample = {}
    for f in filings:
        ftype = f["filing_type"]
        if ftype not in sample:
            sample[ftype] = f
        if len(sample) == 5:
            break

    sample_filings = list(sample.values())

    temperatures = [0.0, 0.3, 0.7, 1.0]

    print("Temperature Experiment Results")
    print("==============================================")
    print("Temp  | Consistency | Avg Tokens | Avg Cost")
    print("----------------------------------------------")

    encoder = tt.encoding_for_model("gpt-4o")

    for temp in temperatures:
        consistent_count = 0
        total_tokens = 0
        total_cost = 0
        total_calls = 0

        for sf in sample_filings:
            results = []

            for _ in range(3):
                classification = classify_filing(
                    sf["content"],
                    temp,
                    client
                )

                results.append(classification.filing_type)

                tokens = len(encoder.encode(sf["content"]))
                total_tokens += tokens

                cost = (tokens / 1_000_000) * PRICING["gpt-4o"]["input"]
                total_cost += cost

                total_calls += 1

            if len(set(results)) == 1:
                consistent_count += 1

        consistency = consistent_count / len(sample_filings)
        avg_tokens = total_tokens / total_calls
        avg_cost = total_cost / total_calls

        print(
            f"{temp:<5} | "
            f"{int(consistency * 100):>5}%      | "
            f"{int(avg_tokens):>6}     | "
            f"${avg_cost:.6f}"
        )
if __name__ == "__main__":
    experiment_runner()