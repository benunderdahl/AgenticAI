from pydantic import BaseModel
import tiktoken
from starter import *
from dotenv import load_dotenv
import time 

DIRECT_PROMPT = """
You are an expert document analyst reviewing SEC filings.
Summarize the document and extract key findings.

Return ONLY a JSON object with this structure:
{
  "key_findings": ["finding 1", "finding 2"],
  "risk_level": "HIGH, MEDIUM, or LOW",
  "regulatory_references": ["Reg S-K Item 303", "SOX 302"],
  "action_items": ["action 1", "action 2"]
}

Do not wrap in markdown. Return only the JSON object.
"""

CHUNK_PROMPT = """
You are an expert document analyst reviewing a section of an SEC filing.
Summarize this chunk and extract key findings from this section only.

Return ONLY a JSON object with this structure:
{
  "key_findings": ["finding 1", "finding 2"],
  "risk_level": "HIGH, MEDIUM, or LOW",
  "regulatory_references": ["Reg S-K Item 303", "SOX 302"],
  "action_items": ["action 1", "action 2"]
}

Do not wrap in markdown. Return only the JSON object.
"""

MERGE_PROMPT = """
You are an expert document analyst merging summaries from multiple chunks
of the same SEC filing into one consolidated summary.

Rules:
- Deduplicate findings — do not repeat the same finding twice
- Take the HIGHEST risk level across all chunks
- Merge and deduplicate regulatory references and action items
- Preserve all unique findings

Return ONLY a JSON object with this structure:
{
  "key_findings": ["finding 1", "finding 2"],
  "risk_level": "HIGH, MEDIUM, or LOW",
  "regulatory_references": ["Reg S-K Item 303", "SOX 302"],
  "action_items": ["action 1", "action 2"]
}

Do not wrap in markdown. Return only the JSON object.
"""

load_dotenv()
encoder = tiktoken.encoding_for_model('gpt-4o')
DIRECT_THRESHOLD = 2000
CHUNK_OVERLAP = 200
ss_folder = Path(__file__).parent / "short-stories"
results = []

def get_short_stories():
    stories = []
    for file in ss_folder.glob("*.txt"):
        stories.append({
            "filename": file.name,
            "content": file.read_text(encoding='utf-8')
        })
    return stories



class DocumentSummary(BaseModel):
    key_findings: list[str]
    risk_level: str
    regulatory_references: list[str] 
    action_items: list[str] 

class ProcessingResult(BaseModel):
    filename: str
    strategy: str               # direct, chunked
    chunks_processed: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    processing_time_s: float
    summary: DocumentSummary | None

class PipelineReport(BaseModel):
    results: list[ProcessingResult]
    total_documents: int
    total_tokens: int
    total_cost_usd: float
    strategy_breakdown: dict[str, int]  # {"direct": 10, "chunked": 5}
    total_processing_time_s: float

# if doc exceeds threshold chunk and return list with chunks that fit model threshold
def chunk_document(content:str, chunk_size: int = 2000) -> list[str]:
    tokens = encoder.encode(content)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP 
    return chunks

# process each chunk individually then merge before returning 
def process_chunked(content: str):
    chunks = chunk_document(content)
    chunk_summaries = []
    tot_in = 0
    tot_out = 0
    for i, chunk in enumerate(chunks):
        chunk_content, in_tok, out_tok = call_llm(CHUNK_PROMPT, chunk)
        tot_in += in_tok
        tot_out += out_tok
        data = parse_json_response(chunk_content)
        chunk_summaries.append(data)
    merge_input = json.dumps(chunk_summaries, indent=2)
    merged_content, in_tok, out_tok = call_llm(MERGE_PROMPT, merge_input)
    tot_in += in_tok
    tot_out += out_tok
    data = parse_json_response(merged_content)
    return DocumentSummary(**data), tot_in, tot_out, len(chunks)

# pass in the doc and we will check total token size of the doc as a whole
def strategy(content: str) -> str:
    # Selecting strategy type for document based on token count 'direct' | 'chunked'
    token_count = len(encoder.encode(content))
    if token_count < DIRECT_THRESHOLD:
        return "direct"
    else:
        return "chunked"

# functino for direct calls to the llm that fit in direct threshold
def process_direct(content: str):
    result, input_tokens, output_tokens = call_llm(DIRECT_PROMPT, content)
    data = parse_json_response(result)
    return DocumentSummary(**data), input_tokens, output_tokens

# processing every single doc to build report 
def process_document(filename: str, content: str) -> ProcessingResult:
    start = time.time()
    selected_strategy = strategy(content)

    if selected_strategy == 'direct':
        summary, in_tok, out_tok = process_direct(content)
        chunks = 1
    else:
        summary, in_tok, out_tok, chunks = process_chunked(content)

    elapsed = time.time() - start
    cost = estimate_cost(in_tok, out_tok)
    return ProcessingResult(
        filename=filename,
        strategy=selected_strategy,
        chunks_processed=chunks,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        processing_time_s=elapsed,
        summary=summary
    )


# build a report to display after prcessing the documents
def build_report(results: list[ProcessingResult]) -> PipelineReport:
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.processing_time_s for r in results)

    strategy_breakdown = {}
    for r in results:
        strategy_breakdown[r.strategy] = strategy_breakdown.get(r.strategy, 0) + 1

    return PipelineReport(
        results=results,
        total_documents=len(results),
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        strategy_breakdown=strategy_breakdown,
        total_processing_time_s=total_time
    )


if __name__ == "__main__":
    filings = load_filings()
    print("STRATEGY SELECTION + CHUNKING")
    print("=" * 40)
    for f in filings:
        result = process_document(f['filename'], f["content"])
        results.append(result)
        token_count = len(encoder.encode(f['content']))
        selected_strategy = strategy(f['content'])
        print(f"  {f['filename']:<40} {token_count:>6,} tokens → {selected_strategy}")
        if selected_strategy == "direct":
            summary, in_tok, out_tok = process_direct(f['content'])
            chunks = 1
            cost = estimate_cost(in_tok, out_tok)
            print(f"  chunks: {chunks} | tokens: {in_tok + out_tok:,} | cost: ${cost:.6f}")
        else:
            summary, in_tok, out_tok, chunks = process_chunked(f['content'])
            print(f"  chunks: {chunks} | tokens: {in_tok + out_tok:,} | cost: ${cost:.6f}")
        cost = estimate_cost(in_tok, out_tok)
        if summary:
            print(f"  risk: {summary.risk_level}")
            print(f"  findings: {summary.key_findings[:2]}")
    report = build_report(results)
    # risk distribution
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        if r.summary:
            risk_counts[r.summary.risk_level] = risk_counts.get(r.summary.risk_level, 0) + 1

    print()
    print("PIPELINE REPORT")
    print("=" * 40)
    print(f"  Documents processed: {report.total_documents}")
    print(f"  Total tokens:        {report.total_tokens:,}")
    print(f"  Total cost:          ${report.total_cost_usd:.4f}")
    print(f"  Total time:          {report.total_processing_time_s:.1f}s")
    strategy_str = " | ".join(f"{k}: {v}" for k, v in report.strategy_breakdown.items())
    print(f"  Strategy breakdown:  {strategy_str}")
    risk_str = " | ".join(f"{k}: {v}" for k, v in risk_counts.items())
    print(f"  Risk distribution:   {risk_str}")