# 04/08 Lab — Prompt Engineering for Enterprise Applications

## Objective

Build a production-grade SEC filing analysis system using the prompt engineering patterns from today's lecture: system prompts, few-shot prompting, chain-of-thought reasoning, and context window management. You will design system prompts that enforce output schemas, build few-shot classification and extraction pipelines, and compare the accuracy and cost tradeoffs of different prompting strategies on real enterprise data.

## Prerequisites

- Python 3.11+ with virtual environment activated
- `openai`, `tiktoken`, `python-dotenv`, `pydantic` packages installed
- Azure OpenAI credentials in your `.env` file:
  - `AZURE_OPENAI_ENDPOINT` — your Azure OpenAI resource endpoint
  - `AZURE_OPENAI_KEY` — your Azure OpenAI API key
  - `AZURE_OPENAI_DEPLOYMENT` (optional, defaults to `gpt-4.1-nano`)
  - `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-12-01-preview`)
- Sample SEC filings from Lab 1 (run `generate_sample_filings.py` if you have not already)
- Familiarity with: system prompts, few-shot prompting, chain-of-thought, context management

---

## Exercise 1: SEC Filing Classifier with Production System Prompts

**Goal:** Build a filing classification pipeline that uses a production-grade system prompt with strict output schema enforcement, decision boundaries, and edge case handling.


1. Create `ex01-system-prompt-classifier.py` in your working directory.

2. Design two system prompts:
   - A **naive prompt**: vague instructions like "classify this filing accurately" with no output schema
   - A **production prompt** that includes:
     - **Explicit role definition** — tell the model exactly what it is and what its job is. Example:
       ```
       You are an SEC filing classification system used by financial analysts.
       Your sole task is to determine the filing type of a given document excerpt
       and return a structured classification.
       ```
     - **Complete taxonomy** — enumerate every valid filing type so the model does not invent categories. Example:
       ```
       Valid filing types (use these exact values):
       - 10-K: Annual report for a full fiscal year
       - 10-Q: Quarterly report for interim fiscal periods
       - 8-K: Current report for material events (officer changes, acquisitions, etc.)
       - DEF 14A: Proxy statement for shareholder meetings and executive compensation
       - S-1: Registration statement for initial public offerings
       - UNKNOWN: Use when the excerpt does not clearly match any of the above
       ```
     - **Exact JSON output schema** — specify field names, types, and allowed values so downstream parsers never break. Example:
       ```
       Respond with ONLY a JSON object matching this schema:
       {
         "filing_type": "10-K",        // one of: 10-K, 10-Q, 8-K, DEF 14A, S-1, UNKNOWN
         "confidence": 0.95,           // float between 0.0 and 1.0
         "evidence": ["phrase from text", ...],  // 1-3 phrases supporting your classification
         "ambiguity_notes": null       // string explaining uncertainty, or null if confident
       }
       ```
     - **Decision rules** — give the model concrete criteria to disambiguate similar types. Example:
       ```
       Classification rules:
       - References to "quarterly period ended" or three-month financials → 10-Q
       - References to "fiscal year ended" or twelve-month financials → 10-K
       - Reports of sudden leadership changes, acquisitions, or bankruptcies → 8-K
       - Discussion of shareholder votes, proxy solicitation, or executive pay → DEF 14A
       - Language about "shares to be registered" or IPO prospectus → S-1
       ```
     - **Edge case handling** — tell the model what to do when it is uncertain instead of letting it guess. Example:
       ```
       If the excerpt is ambiguous or contains signals for multiple filing types,
       set filing_type to "UNKNOWN", confidence below 0.5, and explain the
       ambiguity in ambiguity_notes.
       ```
     - **Negative constraints** — explicitly forbid common failure modes. Example:
       ```
       Do not wrap the JSON in markdown code fences.
       Do not include any text before or after the JSON object.
       Do not add explanations, disclaimers, or commentary.
       ```

3. Define a Pydantic model for the classification output:

```python
class FilingClassification(BaseModel):
    filing_type: str      # 10-K, 10-Q, 8-K, DEF 14A, S-1, UNKNOWN
    confidence: float     # 0.0 to 1.0
    evidence: list[str]   # Phrases from the text supporting the classification
    ambiguity_notes: str | None  # Explain uncertainty, or null
```

   Add validators: `filing_type` must be in the allowed set, `confidence` must be 0.0-1.0.

4. Run both prompts against the same set of SEC filing excerpts from `resources/sec-filings/`. For each filing:
   - Classify with the naive prompt and attempt JSON parsing
   - Classify with the production prompt and parse into the Pydantic model
   - Track whether JSON parsing succeeded

5. Run a batch classification across all filings with the production prompt. Report:
   - Accuracy (predicted vs. expected filing type from the manifest)
   - Classification distribution
   - Low-confidence count (below 50%)
   - Total tokens and cost

**Expected output:**

```
SYSTEM PROMPT COMPARISON: Naive vs. Production
=============================================
  filing-001.txt (expected: 10-K)
    Naive:      10-K         | JSON parseable: NO
    Production: 10-K         | Confidence: 95%

BATCH CLASSIFICATION
====================
  filing-001.txt   10-K       (95%)  CORRECT
  filing-002.txt   10-Q       (92%)  CORRECT
  ...
  Accuracy: 14/15 (93%)
```

**Explanation:** The system prompt is the primary control mechanism for LLM behavior in enterprise applications. A production system prompt functions like a contract — it specifies the exact output format, decision rules, and error-handling behavior. Without this precision, you get unpredictable outputs that break downstream parsers.

---

## Exercise 2: Few-Shot Classification and Extraction Pipeline

**Goal:** Build a pipeline that uses few-shot prompting for both classification and entity extraction, then quantify the accuracy improvement over zero-shot.

1. Create `ex02-few-shot-pipeline.py`.

2. Design few-shot examples for classification. A few-shot example is a pair of **user** and **assistant** messages that you prepend to your messages array, before the real user message. Each pair shows the model an example input and the exact output you expect. Build a list of 6 example pairs:
   - One example for each filing type (10-K, 10-Q, 8-K, DEF 14A, S-1)
   - One UNKNOWN/ambiguous example
   - Choose examples that cover **boundary cases** — filings that could plausibly be multiple types

   Each example should be a short, hand-written filing excerpt (3-5 sentences) paired with the expected JSON output. For instance:

   ```python
   FEW_SHOT_EXAMPLES = [
       {
           "user": (
               "For the fiscal year ended December 31, 2024, the Company reported "
               "total revenues of $12.3B. The following risk factors could materially "
               "affect our business... Item 7: Management's Discussion and Analysis."
           ),
           "assistant": json.dumps({
               "filing_type": "10-K",
               "confidence": 0.95,
               "evidence": ["fiscal year ended", "Item 7: Management's Discussion"],
               "ambiguity_notes": None,
           }),
       },
       # ... one for each filing type, plus one UNKNOWN
   ]
   ```

   > **Tip:** Do NOT use real filings from `sec-filings/` as your few-shot examples — those are your test set. Write short synthetic excerpts that highlight the distinguishing features of each filing type.

3. Implement two classification functions:
   - `classify_zero_shot(text) -> ClassificationResult`
   - `classify_few_shot(text) -> ClassificationResult`

   Both should use the same system prompt and output schema. The only difference is that the few-shot version prepends the example pairs to the messages array:

   ```python
   # Zero-shot messages:
   messages = [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": filing_text},
   ]

   # Few-shot messages — example pairs go between system and the real user message:
   messages = [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": FEW_SHOT_EXAMPLES[0]["user"]},
       {"role": "assistant", "content": FEW_SHOT_EXAMPLES[0]["assistant"]},
       {"role": "user", "content": FEW_SHOT_EXAMPLES[1]["user"]},
       {"role": "assistant", "content": FEW_SHOT_EXAMPLES[1]["assistant"]},
       # ... remaining examples ...
       {"role": "user", "content": filing_text},  # the real input, always last
   ]
   ```

4. Run both approaches on all filings and compare:
   - Accuracy per approach
   - Which specific filings improved with few-shot

5. Add entity extraction using few-shot prompting:
   - Define an `ExtractedEntity` model with fields: `entity_type` (organization, monetary_amount, date, regulation, person), `value`, and `context`
   - Create 1-2 few-shot extraction examples showing how to extract entities from a filing excerpt
   - Run extraction on 3 filings, comparing zero-shot vs. few-shot entity counts and quality

6. Print a comparison report showing accuracy, entity counts, and token usage for each approach.

> **Tip:** The quality of your few-shot examples matters more than the quantity. One well-chosen boundary example teaches the model more than five obvious ones.

**Expected output:**

```
CLASSIFICATION: Zero-Shot vs. Few-Shot
=======================================
  filing-001.txt  expected=10-K  zero=10-K  OK  | few=10-K  OK
  filing-005.txt  expected=8-K   zero=10-K  MISS| few=8-K   OK

  Zero-shot accuracy: 11/15 (73%)
  Few-shot accuracy:  14/15 (93%)
```

**Explanation:** Few-shot examples implicitly communicate classification criteria that are hard to express in rules. They teach the model the boundary between filing types by showing examples of each. For entity extraction, few-shot examples standardize the output format and teach the model which entities to extract and how to present context.

## Exercise 2.5: Few-Shot Cost Analysis

**Goal:** Quantify the token overhead of few-shot prompting and determine whether the accuracy improvement from Exercise 2 justifies the additional cost at scale.

You already have the classification results and token counts from Exercise 2. This exercise is about turning that data into a cost analysis that could inform a real pipeline decision.

1. In your `ex02-few-shot-pipeline.py` (or a new `ex02b-cost-analysis.py` that imports from it), collect the per-filing token usage you tracked in Exercise 2. For each filing, you should have:
   - `input_tokens` and `output_tokens` for the zero-shot call
   - `input_tokens` and `output_tokens` for the few-shot call

2. Compute the per-filing overhead — how many extra input tokens does each few-shot call use compared to its zero-shot counterpart? This should be roughly constant (it is the token cost of your few-shot examples). Verify this by checking that the difference is consistent across filings.

3. Calculate totals and averages across the full corpus:
   - Total input/output tokens for zero-shot vs. few-shot
   - Total cost for each approach (use the `estimate_cost` helper from `starter.py`)
   - Cost per filing for each approach
   - The **overhead multiplier**: `few_shot_total_cost / zero_shot_total_cost`

4. Project the cost difference at enterprise scale. Assume a pipeline that classifies 10,000 filings per month:
   - Monthly cost for zero-shot
   - Monthly cost for few-shot
   - Monthly dollar difference

5. Print a summary report:

```
FEW-SHOT COST ANALYSIS
======================
Per-filing overhead:   +{N} input tokens (the few-shot examples)

                       Zero-Shot       Few-Shot
Corpus input tokens:   12,340          18,960
Corpus output tokens:  1,850           1,870
Corpus total cost:     $0.0030         $0.0040
Cost per filing:       $0.000200       $0.000267
Overhead multiplier:   1.00x           1.33x

PROJECTION @ 10,000 filings/month:
  Zero-shot:  $2.00/month
  Few-shot:   $2.67/month
  Difference: $0.67/month

ACCURACY vs. COST TRADEOFF:
  Zero-shot accuracy: 11/15 (73%)  →  ~2,700 misclassifications/month
  Few-shot accuracy:  14/15 (93%)  →  ~700 misclassifications/month
  Accuracy gain:      +20%  for  +33% cost
```

6. Add a brief comment in your code (2-3 sentences) answering: **Is the cost increase worth it?** Consider that each misclassification in a real compliance pipeline could require human review. If a human analyst costs $75/hour and spends 5 minutes per misclassified filing, what is the cost of those extra misclassifications compared to the few-shot token overhead?

**Explanation:** In production, prompting strategy decisions are not just about accuracy — they are cost/accuracy tradeoffs evaluated at scale. A 33% cost increase sounds significant until you realize it is $0.67/month and prevents 2,000 misclassifications that would each require human review. This kind of back-of-envelope analysis is how engineering teams justify (or reject) more expensive prompting strategies.

---

## Exercise 3: Chain-of-Thought Compliance Document Reviewer

**Goal:** Build a compliance reviewer that uses chain-of-thought prompting to systematically check a filing for regulatory issues, then compare the depth and accuracy of findings vs. direct prompting.


1. Create `ex03-cot-reviewer.py`.

2. Design a compliance scenario. Use the compliance document provided in `resources/compliance-scenario.txt`, or create your own 10-K excerpt that contains multiple planted compliance issues:
   - Missing climate risk disclosure (required for large accelerated filers)
   - Incomplete MD&A (margin decline without trajectory discussion)
   - Late 8-K filing for officer change
   - Thin risk factor disclosure (boilerplate language)
   - Cross-reference inconsistency between sections

3. Implement two review approaches:
   - **Direct prompting**: System prompt asks for compliance issues; returns JSON with issues and severity
   - **Chain-of-thought prompting**: System prompt specifies 5 analysis steps (Reg S-K 303, Reg S-K 105, SOX 302/906, filing timeliness, cross-referencing), then requests JSON output

4. Define Pydantic models:

```python
class ComplianceIssue(BaseModel):
    issue: str
    severity: str       # HIGH, MEDIUM, LOW
    regulation: str     # Specific regulation or section
    evidence: str       # Quote from the filing

class ComplianceAssessment(BaseModel):
    issues: list[ComplianceIssue]
    overall_risk: str   # HIGH, MEDIUM, LOW
    reasoning_steps: list[str] | None = None  # Only present in CoT
```

5. Run both approaches on the same document. Compare:
   - Number of issues found
   - Whether cross-reference issues (the hardest to detect) were caught
   - Token usage and cost
   - Processing time

6. Print a side-by-side comparison with cost analysis.

> **Warning:** CoT uses significantly more output tokens. Track the cost difference — in a production pipeline processing thousands of documents, this adds up. A smart pipeline uses direct prompting for simple documents and CoT only for complex ones.

**Expected output:**

```
DIRECT: 3 issues found | $0.000045 | 1.2s
COT:    6 issues found | $0.000180 | 2.8s
Cost multiplier: 4.0x

Issues found only by CoT:
  - [HIGH] Late 8-K filing (cross-reference with officer change)
  - [MEDIUM] Interim CFO certification tenure concern
  - [LOW] Missing AI/ML governance risk factor
```

**Explanation:** Chain-of-thought prompting forces the model to systematically work through each compliance dimension before rendering a judgment. This catches cross-reference issues that direct prompting misses because the model has already "written down" its findings about each section before looking for inconsistencies. The tradeoff is 3-5x more output tokens.

---

## Exercise 4: Context-Managed Document Processing Pipeline

**Goal:** Build a pipeline that automatically selects a processing strategy (direct or chunked) based on document size, tracks cost across the batch, and produces a consolidated report.


1. Create `ex04-context-pipeline.py`.

2. Implement token-aware strategy selection:
   - If document is under 6,000 tokens: process directly (single LLM call)
   - If document is 6,000+ tokens: chunk the document, process each chunk, then merge results

3. Build a chunking function:
   - Split text into chunks of 3,000 tokens maximum
   - Overlap chunks by 200 tokens to avoid losing context at boundaries
   - Use `tiktoken` for accurate token-based splitting (not character-based)

4. Implement the processing pipeline:
   - **For direct documents**: single LLM call to summarize and extract findings
   - **For chunked documents**: map (summarize each chunk) then reduce (merge summaries)
   - Define a `DocumentSummary` Pydantic model with: `key_findings`, `risk_level`, `regulatory_references`, `action_items`

5. Track metrics per document:
   - Strategy used (direct/chunked)
   - Number of chunks processed
   - Input/output tokens
   - Cost
   - Processing time

6. Run on all SEC filings. Produce a pipeline report:

```python
class PipelineReport(BaseModel):
    results: list[ProcessingResult]
    total_documents: int
    total_tokens: int
    total_cost_usd: float
    strategy_breakdown: dict[str, int]  # {"direct": 10, "chunked": 5}
    total_processing_time_s: float
```

> **Tip:** The merge step is the hardest part. When combining chunk summaries, you need to deduplicate findings, take the highest risk level across chunks, and merge regulatory references. Design the merge prompt carefully.

**Expected output:**

```
PIPELINE REPORT
===============
  Documents processed: 15
  Total tokens:        28,450
  Total cost:          $0.0043
  Total time:          34.2s
  Strategy breakdown:  direct: 10 | chunked: 5
  Risk distribution:   HIGH: 3 | MEDIUM: 7 | LOW: 5
```

**Explanation:** Real enterprise document processing pipelines must handle documents of wildly different sizes — from 1-page memos to 200-page contracts. Automatic strategy selection based on token count is the baseline capability. The map-reduce pattern for chunked processing is the same pattern used in production RAG systems (which you will build in Week 03).

---

## Stretch Goals

### Stretch 1: Prompt Optimization Benchmark

Build a systematic benchmark that tests 4 variations of your classification system prompt against the full filing corpus:
1. Minimal prompt (role + task only)
2. Prompt with output schema
3. Prompt with schema + decision rules
4. Full production prompt (schema + rules + edge cases + negative constraints)

For each variation, measure: accuracy, JSON parse success rate, average confidence, token usage, and cost. Produce a comparison table. This gives you empirical data on which prompt engineering techniques provide the most value for the cost.

### Stretch 2: Self-Consistency Voting

Implement the self-consistency technique from the lecture: run the same CoT compliance review prompt 5 times at temperature 0.3, then take the majority vote for each issue. Compare the accuracy of the majority-voted result vs. a single CoT run vs. a single direct run. Track the total cost (5x the single-run cost) and determine whether the accuracy improvement justifies the expense.

---

## Troubleshooting Tips

- **LLM returns markdown code fences around JSON**: Strip the ```json prefix and trailing ``` before parsing. This is common when the model "helpfully" formats its output.

- **Pydantic ValidationError on severity field**: The LLM may return "High" instead of "HIGH". Add a `@field_validator` that normalizes to uppercase, or use `Literal["HIGH", "MEDIUM", "LOW"]` with a pre-validator.

- **Few-shot examples consuming too many tokens**: Each few-shot example consumes prompt tokens on every call. If you have 5 long examples, that is 2,000-3,000 extra tokens per call. Use concise examples — the filing excerpts in examples should be 3-5 sentences, not full pages.

- **CoT output too long to parse**: The model may generate extensive reasoning that overflows `max_tokens` before reaching the JSON. Increase `max_tokens` for CoT calls (1500-2000), or restructure the prompt to put reasoning steps inside the JSON object rather than before it.

- **Chunked processing misses cross-chunk entities**: An entity mentioned on chunk boundary may be split. The 200-token overlap mitigates this, but some information loss is inherent. For production, consider a sliding window with larger overlap or a two-pass approach.

- **Token counts differ between tiktoken and API response**: The API's token count includes per-message overhead (role tokens, delimiters) that tiktoken on raw text does not. Budget an extra ~4 tokens per message.