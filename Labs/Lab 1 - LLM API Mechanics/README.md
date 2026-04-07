# 04/07 Lab — LLM API Mechanics, Tokenization, and Cost-Aware Engineering

## Objective

Build a production-grade SEC filing analysis system using raw LLM API calls. You will make direct API calls (no frameworks), count tokens, manage context windows, implement retry logic, and track costs. This lab simulates the engineering challenges of building real-world LLM applications in an enterprise context.

## Prerequisites

- Python 3.11+ with virtual environment activated

---

## Setup

Complete these steps before starting any exercise. They only need to be done once.

### 1. Create and activate a virtual environment

Virtual environments isolate your project's dependencies from the system Python and from other projects on the same machine. In enterprise settings this is critical — teams often maintain multiple services pinned to different library versions, and a global `pip install` can silently break another project's dependency tree. Virtual environments also make reproducible deployments possible: the `requirements.txt` inside a venv-backed project is the exact manifest that goes into your CI/CD pipeline and production containers.

```bash
# Create the virtual environment (only needed once per project)
python -m venv .venv

# Activate it
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

> **Tip:** You will know the environment is active when your terminal prompt is prefixed with `(.venv)`. You need to re-activate every time you open a new terminal session.

### 2. Install required packages

```bash
pip install openai tiktoken python-dotenv pydantic
```

### 3. Create your `.env` file

In the `resources/` directory (same folder as `starter.py`), create a file named `.env`:
In that file, put:
```
OPENAI_API_KEY=sk-your-key-here
```

Get your key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys). Never commit this file to git.

### 4. Generate sample SEC filings

The exercises use realistic synthetic SEC filings. Run this once to create the data:

```bash
cd resources/
python generate_sample_filings.py
```

You should see output like:
```
Generated 28 SEC filing excerpts in .../resources/sec-filings/
Manifest written to .../resources/sec-filings/manifest.json
  01-meridian-technologies-inc-10k.txt: 10-K | Meridian Technologies Inc. | 12,450 chars
  02-cascade-financial-group-8k.txt: 8-K | Cascade Financial Group | 1,230 chars
  ...
```

### 5. Verify your setup

```bash
python starter.py
```

Expected output: `Loaded 28 filings from .../resources/sec-filings/`

If you see `OPENAI_API_KEY not found`, check that your `.env` file is in the `resources/` directory.

### Understanding the starter file

Open `starter.py` — it provides utilities you should import in your exercise files (which you will create in the same `resources/` folder):

```python
from starter import load_all_filings, load_filing, DATA_DIR
```

`load_all_filings()` returns a list of dicts, each with keys: `filename`, `company`, `sector`, `cik`, `filing_type`, `char_count`, and `content` (the full text).

---

## OpenAI SDK Primer

Before diving into exercises, here are the key SDK patterns you'll use throughout the lab. Read this section in full — it covers the non-obvious parts.

### Initializing the client

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Making a chat completion call

Every call passes a `messages` list. Each message is a dict with `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (the text):

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Regulation S-K?"},
    ],
    temperature=0.0,
    max_tokens=500,
)
```

### Reading the response

The response is a structured object, not a dict. Access it like this:

```python
# The model's reply text
reply_text = response.choices[0].message.content

# Token usage (always present)
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens
```

### Forcing JSON output

When you need the model to return valid JSON (Exercises 2, 4), use `response_format`:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={"type": "json_object"},  # Forces valid JSON output
    temperature=0.0,
)

import json
data = json.loads(response.choices[0].message.content)
```

> **Warning:** `response_format={"type": "json_object"}` requires your system prompt or user message to mention the word "JSON" — otherwise the API returns an error. A system prompt like `"Respond in valid JSON only."` satisfies this.

### Catching API errors

The openai library raises typed exceptions. The most important ones:

```python
from openai import RateLimitError, APIStatusError, APIConnectionError

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    # HTTP 429 — you've exceeded your rate limit, retry after a delay
    pass
except APIStatusError as e:
    # HTTP 4xx/5xx — e.status_code gives you the code
    # 400/401/403 are client errors — don't retry these
    # 500/503 are server errors — safe to retry
    print(e.status_code, e.message)
except APIConnectionError:
    # Network failure — safe to retry
    pass
```

---

## Exercise 1: Token-Aware Document Analyzer

**Goal:** Build a utility that analyzes SEC filings for token consumption, estimates API costs, and recommends a processing strategy for each document.


### Steps

1. Create a file `token_analyzer.py` in the `resources/` folder.

2. At the top of the file, import tiktoken and define your pricing constants. Use these values — they reflect current OpenAI list prices per 1M input tokens:

```python
import tiktoken

PRICING = {
    "gpt-4o":             {"input": 2.50, "output": 10.00},   # USD per 1M tokens
    "claude-3.5-sonnet":  {"input": 3.00, "output": 15.00},
}

MODEL_LIMITS = {
    "gpt-4o":            {"context_window": 128_000},
    "claude-3.5-sonnet": {"context_window": 200_000},
}

SYSTEM_PROMPT_RESERVE = 2_000   # tokens reserved for your system prompt
OUTPUT_RESERVE        = 1_000   # tokens reserved for model's reply
```

3. Initialize the tiktoken encoder for GPT-4o. Tiktoken is a tokenizer — it splits text into the same tokens the model uses, so your counts are exact:

```python
encoder = tiktoken.encoding_for_model("gpt-4o")
token_count = len(encoder.encode(some_text))
```

4. Define the `DocumentStats` Pydantic model:

```python
from pydantic import BaseModel

class DocumentStats(BaseModel):
    filename: str
    char_count: int
    token_count: int
    chars_per_token: float
    fits_gpt4o: bool           # True if tokens fit within gpt-4o's usable window
    fits_claude_sonnet: bool   # True if tokens fit within Claude's usable window
    estimated_cost_gpt4o: float
    estimated_cost_claude: float
    recommended_strategy: str  # "direct", "chunked", or "rag_required"
```

5. Implement a `DocumentAnalyzer` class with an `analyze_all()` method that:
   - Loads all filings using `load_all_filings()` from `starter.py`
   - Counts tokens for each filing using tiktoken
   - Calculates `fits_gpt4o` / `fits_claude_sonnet`: subtract `SYSTEM_PROMPT_RESERVE + OUTPUT_RESERVE` from each model's context window — does the document fit in the remaining space?
   - Estimates cost: `(token_count / 1_000_000) * price_per_million`. Use the input price only (you are analyzing, not generating long output).
   - Assigns `recommended_strategy`:
     - `"direct"` — under 50,000 tokens
     - `"chunked"` — 50,000 to 200,000 tokens
     - `"rag_required"` — over 200,000 tokens

6. Add a `print_report()` method that produces output in this format:

```
Document Analysis Report
========================
  01-meridian-technologies-10k.txt:  12,450 tokens  |  $0.0000  |  direct
  02-cascade-financial-8k.txt:        1,230 tokens  |  $0.0000  |  direct
  ...

Summary: 28 documents | 87,400 total tokens | $0.0002 total est. cost (gpt-4o)
Strategy breakdown: 25 direct | 3 chunked | 0 rag_required
```

7. In your `if __name__ == "__main__":` block, instantiate the analyzer and call `print_report()`.

**Expected output:** All or nearly all filings should fall into the `"direct"` category, since the synthetic filings are short. A few padded 10-K filings may fall into `"chunked"`.

**Explanation:** Token counting is not optional in enterprise LLM systems. Every API call has a cost, a context limit, and a latency profile — all driven by token count. Building this awareness into your tools from the start prevents cost surprises and context window errors in production.

---

## Exercise 2: SEC Filing Classifier with Temperature Experiments

**Goal:** Build a filing classifier and empirically measure how temperature affects classification consistency.


### Steps

1. Create `filing_classifier.py` in the `resources/` folder.

2. Define the output model:

```python
from pydantic import BaseModel

class FilingClassification(BaseModel):
    filing_type: str    # Expected values: 10-K, 10-Q, 8-K, DEF 14A
    confidence: float   # 0.0 to 1.0
    company_name: str
    primary_sector: str
    reasoning: str      # One sentence explaining the classification
```

3. Implement a `classify_filing(text: str, temperature: float, client: OpenAI) -> FilingClassification` function. It should:
   - Build a messages array with a system prompt that instructs the model to classify SEC filings and respond in JSON
   - Pass `response_format={"type": "json_object"}` to guarantee parseable output
   - Call `client.chat.completions.create()` with `model="gpt-4o-mini"`, `max_tokens=200`
   - Parse the JSON response and validate it with your Pydantic model:

```python
import json
from pydantic import ValidationError

raw = response.choices[0].message.content
try:
    data = json.loads(raw)
    return FilingClassification(**data)
except (json.JSONDecodeError, ValidationError) as e:
    # Retry once with a stricter prompt before giving up
    ...
```

   - Log the token usage for each call (print to console is fine for this exercise)

4. Implement an `experiment_runner()` function that:
   - Loads all filings using `load_all_filings()`
   - Picks a representative sample of 5 filings (one of each type if possible — use `filing_type` from the manifest dict, not from the model output)
   - For each temperature in `[0.0, 0.3, 0.7, 1.0]`:
     - For each of the 5 filings, calls `classify_filing()` 3 times
     - Tracks whether `filing_type` is consistent across all 3 runs (all the same = consistent)
   - Calculates consistency: `(number of filings with all 3 runs matching) / 5` per temperature

5. Print a results table:

```
Temperature Experiment Results
==============================
Temp  | Consistency | Avg Tokens | Avg Cost
0.0   |     100%    |    145     |  $0.0001
0.3   |      80%    |    152     |  $0.0001
0.7   |      60%    |    168     |  $0.0001
1.0   |      40%    |    189     |  $0.0002
```

> **Tip:** The `manifest.json` written by `generate_sample_filings.py` stores the ground-truth `filing_type` for each document. Use `load_all_filings()` which returns this in each entry's `"filing_type"` key. This lets you also check whether the model's classification matches the ground truth.

> **Warning:** This experiment makes up to 60 API calls (5 filings × 4 temperatures × 3 runs). With `gpt-4o-mini` at short inputs, cost should be under $0.01 total. Verify before running.

**Expected behavior:** Temperature 0.0 should produce 100% (or near-100%) consistency. Higher temperatures should show decreasing consistency, especially in the `reasoning` and `primary_sector` fields. The `filing_type` field may remain stable until temperature 1.0 because filing type classification is highly constrained.

**Explanation:** Enterprise systems require deterministic behavior for classification, extraction, and validation tasks. This exercise demonstrates empirically why temperature 0 is the default for these use cases. Higher temperatures generate more diverse (and more expensive) output tokens.

---

## Exercise 3: Multi-Turn Compliance Q&A with Context Management

**Goal:** Build a multi-turn Q&A system that tracks conversation context, manages the growing message array, and implements a sliding window when the context approaches its limit.


### Steps

1. Create `compliance_qa.py` in the `resources/` folder.

2. Define the response and stats models:

```python
from pydantic import BaseModel

class QAResponse(BaseModel):
    answer: str
    turn_number: int
    input_tokens: int
    output_tokens: int
    total_tokens_in_context: int
    context_was_trimmed: bool
    cost_usd: float

class SessionStats(BaseModel):
    total_turns: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    trimming_events: int
```

3. Implement a `ComplianceQASession` class:

```python
class ComplianceQASession:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_tokens: int = 12_000,  # Intentionally low to trigger windowing
        max_output_tokens: int = 500,
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        self.turn_number = 0
        self.trimming_events = 0
        # Track cumulative token usage for SessionStats
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # The message history — starts with just the system prompt
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an expert SEC compliance analyst with deep knowledge of "
                    "Regulation S-K, Sarbanes-Oxley (SOX), and SEC reporting requirements. "
                    "Answer questions precisely, citing specific regulations and sections "
                    "where applicable. Keep answers under 200 words."
                ),
            }
        ]

    def ask(self, question: str) -> QAResponse:
        ...

    def get_session_stats(self) -> SessionStats:
        ...
```

4. In the `ask()` method, implement the following sequence:

   **a. Append the new user message:**
   ```python
   self.messages.append({"role": "user", "content": question})
   ```

   **b. Count current tokens in the full message list.** You need to tokenize each message and sum them. A simple approach:
   ```python
   def _count_messages_tokens(self) -> int:
       total = 0
       for msg in self.messages:
           total += len(self.encoder.encode(msg["content"]))
           total += 4  # OpenAI overhead per message (role + formatting tokens)
       total += 2  # Priming tokens OpenAI adds to every request
       return total
   ```

   **c. If total tokens exceed `max_context_tokens - max_output_tokens`, trim the message history.** Important rules:
   - Never remove `self.messages[0]` — that is the system prompt
   - Remove the oldest user/assistant pairs (messages at index 1 and 2) until you are back under the limit
   - Log how many messages were removed and how many tokens were freed
   - After trimming, insert a brief system message summarizing what was cut:
     ```python
     self.messages.insert(1, {
         "role": "system",
         "content": "Note: earlier conversation context was trimmed to manage context window size.",
     })
     ```

   **d. Make the API call:**
   ```python
   response = self.client.chat.completions.create(
       model=self.model,
       messages=self.messages,
       temperature=0.0,
       max_tokens=self.max_output_tokens,
   )
   ```

   **e. Append the assistant's reply to the history** (so the next turn sees it):
   ```python
   self.messages.append({
       "role": "assistant",
       "content": response.choices[0].message.content,
   })
   ```

   **f. Build and return the `QAResponse`** using `response.usage.prompt_tokens`, `response.usage.completion_tokens`, and the current `_count_messages_tokens()` value.

5. Write a test script that asks at least 8 sequential compliance questions that build on each other. Example sequence:

```python
questions = [
    "What are the core components required in a 10-K annual report under Regulation S-K?",
    "Which specific items under Regulation S-K govern executive compensation disclosure?",
    "How does Item 402 of Regulation S-K define 'named executive officers'?",
    "What did SOX Section 302 add to the CEO/CFO certification requirements?",
    "How does SOX Section 906 differ from Section 302 in terms of penalties?",
    "Can a company file an amended 10-K (10-K/A) to correct executive compensation disclosures?",
    "What is the deadline for a large accelerated filer to file its 10-K?",
    "If a company misses the 10-K deadline, what are the SEC's immediate enforcement options?",
]
```

6. After all turns, print a summary showing token count per turn so you can see the sawtooth pattern (growth → trim → growth again).

**Expected behavior:** With `max_context_tokens=12_000`, trimming should occur around turn 4-6 depending on answer length. You should see `context_was_trimmed=True` on at least one response, and the `total_tokens_in_context` value should drop after trimming.

**Explanation:** Context window management is one of the most critical engineering problems in LLM applications. Every multi-turn system must handle the case where conversations exceed the context limit. The sliding window is the simplest approach — later in the program you will learn summarization-based compression and retrieval-augmented alternatives.

---

## Exercise 4: Production API Client with Retry and Cost Tracking

**Goal:** Build a reusable, production-grade LLM API client that wraps the OpenAI SDK with retry logic, cost tracking, input validation, and structured logging.


### Steps

1. Create `enterprise_llm_client.py` in the `resources/` folder.

2. Define the supporting types:

```python
import time
import logging
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    use_case: str
    cost_usd: float


class CostReport(BaseModel):
    total_calls: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    by_use_case: dict[str, dict]  # {"classification": {"calls": N, "cost": X}, ...}


class BudgetExceededError(Exception):
    """Raised when cumulative spend would exceed the configured budget."""
    pass
```

3. Implement the `EnterpriseLLMClient` class. Implement each feature one at a time:

```python
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # USD per 1M tokens
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

class EnterpriseLLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        budget_limit_usd: float = 5.0,
        max_retries: int = 3,
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.budget_limit_usd = budget_limit_usd
        self.max_retries = max_retries
        self.encoder = tiktoken.encoding_for_model(model)
        self._call_log: list[LLMResponse] = []   # every successful call goes here
        self._cumulative_cost: float = 0.0

    def complete(
        self,
        messages: list[dict[str, str]],
        use_case: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        ...

    def get_cost_report(self) -> CostReport:
        ...
```

4. Implement `complete()` with these four features in order:

   **a. Pre-call token validation.**
   Count tokens in the messages list before making the call. If the count exceeds the model's context window (use 128,000 for gpt-4o-mini), raise a `ValueError` immediately — don't spend the API call to find out.

   **b. Budget enforcement.**
   Before making the call, calculate what this call would cost (estimate: current message tokens × input price per token). If `_cumulative_cost + estimated_cost > budget_limit_usd`, raise `BudgetExceededError` with a helpful message showing how much was spent and what the limit is.

   **c. Retry with exponential backoff.**
   Wrap the API call in a retry loop. Use `time.sleep()` with jitter:
   ```python
   for attempt in range(self.max_retries):
       try:
           start = time.time()
           response = self.client.chat.completions.create(...)
           latency_ms = (time.time() - start) * 1000
           break  # success — exit the retry loop
       except RateLimitError:
           if attempt == self.max_retries - 1:
               raise
           wait = (2 ** attempt) + random.uniform(0, 1)  # exponential + jitter
           logger.warning(f"Rate limited. Retry {attempt + 1}/{self.max_retries} in {wait:.1f}s")
           time.sleep(wait)
       except APIStatusError as e:
           if e.status_code in (400, 401, 403):
               raise  # client errors — never retry
           if attempt == self.max_retries - 1:
               raise
           wait = (2 ** attempt) + random.uniform(0, 1)
           logger.warning(f"Server error {e.status_code}. Retry {attempt + 1}/{self.max_retries}")
           time.sleep(wait)
       except APIConnectionError:
           if attempt == self.max_retries - 1:
               raise
           time.sleep(2 ** attempt)
   ```

   **d. Log the result and track cost.**
   After a successful call, calculate actual cost from `response.usage`, build an `LLMResponse`, append it to `_call_log`, update `_cumulative_cost`, and log at INFO level:
   ```python
   logger.info(f"{use_case} | {response.usage.total_tokens} tokens | ${cost:.4f} | {latency_ms:.0f}ms")
   ```

5. Implement `get_cost_report()` by aggregating `_call_log`. Group by `use_case`, sum costs and calls per group.

6. Test the client with the SEC filings:
   - Load all filings with `load_all_filings()`
   - Build a simple classification messages array for each filing (system prompt + the filing content as the user message)
   - Call `complete()` with `use_case="classification"` for each filing
   - Then call again with `use_case="extraction"` asking to extract company name and CIK
   - Print `get_cost_report()` after all calls

7. Test budget enforcement separately: create a second client with `budget_limit_usd=0.0001` and verify `BudgetExceededError` is raised before the second call completes.

**Expected output:**

```
INFO | classification | 145 tokens | $0.0000 | 312ms
INFO | classification | 152 tokens | $0.0000 | 298ms
...

Cost Report
===========
  Total calls:    56
  Total tokens:   8,450
  Total cost:     $0.0013
  Avg latency:    310ms
  By use case:
    classification:  $0.0004  (28 calls)
    extraction:      $0.0009  (28 calls)

Budget test:
BudgetExceededError: Budget limit $0.0001 reached after $0.0001 spent
```

> **Tip:** The `BudgetExceededError` test will fire very quickly — after just 1-2 calls — because the budget is intentionally tiny. That is the expected behavior.

**Explanation:** This client is the pattern you will reuse throughout the program. Production LLM systems require the same engineering rigor as any other external service integration — retries, cost controls, and observability. The patterns here (exponential backoff with jitter, pre-validation, budget circuits) are directly applicable to real enterprise systems.

---

## Stretch Goals

### Stretch 1: Async Batch Processing with Concurrency Control

Extend your `EnterpriseLLMClient` to support async batch processing. Use `asyncio` and `openai.AsyncOpenAI` with `asyncio.Semaphore` to process N documents simultaneously while respecting rate limits. Measure throughput (documents/minute) at concurrency levels 1, 3, 5, and 10. Plot the results with `matplotlib`.

Note: `AsyncOpenAI` uses `await client.chat.completions.create(...)` instead of the synchronous version. The response object is identical.

### Stretch 2: Token Budget Optimizer

Given all 28 filings and a total token budget (e.g., 50,000 tokens), determine the optimal processing strategy for each document — `direct`, `chunked`, or `skip` — to maximize the number of documents fully processed within the budget. This is a constrained optimization problem. Consider a greedy approach (sort by token count, process cheapest first) or a dynamic programming approach. Compare the two strategies' outcomes.

---

## Troubleshooting Tips

- **`ModuleNotFoundError: No module named 'openai'`**: Your virtual environment is not activated. Run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows).

- **`AuthenticationError`**: Your `OPENAI_API_KEY` is missing or invalid. Check your `.env` file in the `resources/` folder. Verify the key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

- **`FileNotFoundError: Manifest not found`**: You haven't generated the sample data yet. Run `python generate_sample_filings.py` from the `resources/` directory.

- **`RateLimitError` on first call**: You may have exceeded your account's tier limits. Check your usage dashboard at [https://platform.openai.com/usage](https://platform.openai.com/usage).

- **Token counts differ from what you expect**: Different models use different tokenizers. Always use `tiktoken.encoding_for_model("gpt-4o-mini")` with the exact model name rather than a generic encoding like `cl100k_base` — the model name ensures the right tokenizer version.

- **`BadRequestError: 'response_format' must include 'json'`**: When using `response_format={"type": "json_object"}`, your system prompt or user message must contain the word "JSON". Add `"Respond in valid JSON."` to your system prompt.

- **Pydantic `ValidationError` from LLM output**: The model returned a JSON field with an unexpected value (e.g., `"HIGH"` instead of `"High"`). Use `model_config = ConfigDict(str_to_lower=True)` or add a Pydantic `field_validator` to normalize before validation.

- **Sliding window not triggering in Exercise 3**: The default `max_context_tokens=12_000` is deliberately low to force trimming. If trimming isn't triggering, check that your `_count_messages_tokens()` method is being called before every API call and that the comparison uses `max_context_tokens - max_output_tokens` as the threshold.

- **Cost is $0.0000 for every call**: You are using `gpt-4o-mini` on short documents — the cost is real but rounds to zero at 4 decimal places. Increase `max_tokens` or use more documents to see non-zero values.
