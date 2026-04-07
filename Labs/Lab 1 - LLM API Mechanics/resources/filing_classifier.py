from pydantic import BaseModel, ValidationError
import json
from openai import OpenAI
from starter import load_all_filings
import tiktoken as tt


class FilingClassification(BaseModel):
    filing_type: str    # Expected values: 10-K, 10-Q, 8-K, DEF 14A
    confidence: float   # 0.0 to 1.0
    company_name: str
    primary_sector: str
    reasoning: str      # One sentence explaining the classification
   

   
def classify_filing(text: str, temperature: float, client: OpenAI) -> FilingClassification:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify SEC filings and return ONLY JSON with:\n"
                "filing_type, confidence, company_name, primary_sector, reasoning"
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    client = OpenAI()
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
            f"{int(consistency * 100):>5}%       | "
            f"{int(avg_tokens):>6}     | "
            f"${avg_cost:.6f}"
        )
if __name__ == "__main__":
    experiment_runner()