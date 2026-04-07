import tiktoken as tt
from openai import RateLimitError, APIStatusError, APIConnectionError
from pydantic import BaseModel
from starter import load_all_filings
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

class DocumentAnalyzer:
    
    def analyze_all(self):
            try: 
                encoder = tt.encoding_for_model("gpt-4o")
                filings = load_all_filings()
                results = []
                for file in filings:
                    filename = file['filename']
                    char_count = len(file['content'])
                    token_count = len(encoder.encode(file['content']))
                    chars_per_token = char_count / token_count if token_count > 0 else 0
                    fits_gpt40 = token_count <= MODEL_LIMITS["gpt-4o"]['context_window'] - SYSTEM_PROMPT_RESERVE + OUTPUT_RESERVE
                    fits_claude_sonnet = token_count <= MODEL_LIMITS["claude-3.5-sonnet"]['context_window'] - SYSTEM_PROMPT_RESERVE + OUTPUT_RESERVE
                    estimated_cost_gpt40 = (token_count / 1_000_000) * PRICING["gpt-4o"]["input"]
                    estimated_cost_claude = (token_count / 1_000_000) * PRICING["claude-3.5-sonnet"]["input"]
                    if fits_gpt40:
                        recommended_strategy = "direct"
                    elif token_count>= 50_000 and token_count<= 200_000:
                        recommended_strategy = "chunked"
                    else:
                        recommended_strategy = "rag_required"

                    stats = DocumentStats(
                        filename=filename,
                        char_count=char_count,
                        token_count=token_count,
                        chars_per_token=chars_per_token,
                        fits_gpt4o=fits_gpt40,
                        fits_claude_sonnet=fits_claude_sonnet,
                        estimated_cost_gpt4o=estimated_cost_gpt40,
                        estimated_cost_claude=estimated_cost_claude,
                        recommended_strategy=recommended_strategy
                    )
                    results.append(stats)
                return results
            except(RateLimitError, APIStatusError, APIConnectionError) as e:
                print(e)
            
    def print_report(self, results):
        total_tokens = 0
        total_est_cost = 0
        direct = 0
        chunked = 0
        rag_required = 0
        print(f"{"Document Analysis Report":>60}")
        print("=" * 120)
        for result in results:
            print(f"{result.filename:<50}:     {result.token_count: >10} tokens   |      ${round(result.estimated_cost_gpt4o,4):>10}     |     {result.recommended_strategy :>10}")
            total_tokens += result.token_count
            total_est_cost += result.estimated_cost_gpt4o
            if result.recommended_strategy == 'direct':
                direct += 1
            elif result.recommended_strategy == 'chunked':
                chunked += 1
            else:
                rag_required += 1
        print(f"Summary: {len(results)} documents | {total_tokens} tokens | {round(total_est_cost, 4)} total est. cost (gpt-4o)")
        print(f"Strategy Breakdown: {direct} direct | {chunked} chunked | {rag_required} rag_required")
            
if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    analyzer.print_report(analyzer.analyze_all())