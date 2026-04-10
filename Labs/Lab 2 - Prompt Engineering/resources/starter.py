"""
04/08 Lab Starter — Prompt Engineering for Enterprise Applications

The sample SEC filings from Lab 01 are reused. Run
generate_sample_filings.py (in Lab 01 - LLM API Mechanics/resources/) if you have not already.
"""

import json
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Configuration ---

DATA_DIR = Path(__file__).parent  / "sec-filings"
DATA_DIR = DATA_DIR.resolve()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
    raise EnvironmentError(
        "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY not found. "
        "Create a .env file with your Azure OpenAI credentials."
    )

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

# Pricing per 1M tokens for gpt-4.1-nano
PRICING_INPUT = 0.10
PRICING_OUTPUT = 0.40


# --- Data Loading ---

def load_filings() -> list[dict]:
    """Load all SEC filing excerpts with metadata."""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run day-01/resources/generate_sample_filings.py first."
        )
    filings = json.loads(manifest_path.read_text(encoding="utf-8"))
    for f in filings:
        filepath = DATA_DIR / f["filename"]
        f["content"] = filepath.read_text(encoding="utf-8")
    return filings


# --- Token Utilities ---

def count_tokens(text: str) -> int:
    """Count tokens in a string using the gpt-4o-mini tokenizer."""
    return len(encoder.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single API call."""
    return (
        (input_tokens / 1_000_000) * PRICING_INPUT
        + (output_tokens / 1_000_000) * PRICING_OUTPUT
    )


# --- LLM Call Helper ---

def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> tuple[str, int, int]:
    """Make an LLM call and return (content, input_tokens, output_tokens)."""
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return (
        content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

def call_llm_few(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    extra_messages: list[dict] | None = None,  # ← add this
) -> tuple[str, int, int]:
    """Make an LLM call and return (content, input_tokens, output_tokens)."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if extra_messages:
        messages.extend(extra_messages)  
        
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return (
        content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

def parse_json_response(content: str) -> dict | None:
    """Parse LLM output as JSON, stripping markdown fences if present."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # Strip ```json or ``` prefix and trailing ```
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# --- Quick verification ---

if __name__ == "__main__":
    try:
        filings = load_filings()
        print(f"Loaded {len(filings)} filings from {DATA_DIR}")
        for f in filings[:5]:
            tokens = count_tokens(f["content"])
            print(
                f"  {f['filename']}: {f['filing_type']} | "
                f"{tokens:,} tokens | "
                f"${estimate_cost(tokens, 200):.6f} est."
            )
        if len(filings) > 5:
            print(f"  ... and {len(filings) - 5} more")
    except FileNotFoundError as e:
        print(f"Setup needed: {e}")
