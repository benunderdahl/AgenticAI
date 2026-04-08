from pydantic import BaseModel, ValidationError
import os
from openai import AzureOpenAI, RateLimitError, APIStatusError, APIConnectionError
import tiktoken
import time
import logging
from dotenv import load_dotenv
from filing_classifier import load_all_filings


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


class EnterpriseLLMClient:
    PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # USD per 1M tokens
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    }
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        budget_limit_usd: float = 5.0,
        max_retries: int = 3,
    ):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION")
            )
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
        total_tokens = sum(len(self.encoder.encode(msg['content'])) + 4 for msg in messages)
        context_limit = 128_000
        if total_tokens > context_limit:
            raise ValueError(f"Token count {total_tokens} exceeds context window limit of {context_limit}")
        estimated_cost = (total_tokens / 1_000_000) * self.PRICING['gpt-4o-mini']['input']
        if self._cumulative_cost + estimated_cost > self.budget_limit_usd:
            raise BudgetExceededError(f"Budget exceeded - spent ${self._cumulative_cost} of " + 
                f"${self.budget_limit_usd} limit estimated cost would be ${estimated_cost}")
        try:
            total_tokens=0
            cumulative_cost=0
            for msg in messages:
                total_tokens += len(self.encoder.encode(msg['content']))
                cumulative_cost += len(self.encoder.encode(msg['content'])) * self.PRICING['gpt-4o-mini']['input']
                if total_tokens > 128_000:
                    raise ValueError("Too many tokens")
                elif cumulative_cost > self.budget_limit_usd:
                    raise BudgetExceededError(f"")
        except (ValueError, BudgetExceededError) as e:
            print(e)
        
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens
                )
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
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            actual_cost = (
                (input_tokens  / 1_000_000) * self.PRICING["gpt-4o-mini"]["input"] +
                (output_tokens / 1_000_000) * self.PRICING["gpt-4o-mini"]["output"]
                )
            self._cumulative_cost += actual_cost

            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model=self.model,
                use_case=use_case,
                cost_usd=actual_cost
            )

            self._call_log.append(llm_response)

            logger.info(
                f"{use_case} | {response.usage.total_tokens} tokens | "
                f"${actual_cost:.4f} | {latency_ms:.0f}ms"
                )
            return llm_response
        
    def get_cost_report(self) -> CostReport:
            if not self._call_log:
                return CostReport(
                    total_calls=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    avg_latency_ms=0.0,
                    by_use_case={}
                )

            total_calls = len(self._call_log)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in self._call_log)
            total_cost = sum(r.cost_usd for r in self._call_log)
            avg_latency = sum(r.latency_ms for r in self._call_log) / total_calls

            # group by use_case
            by_use_case = {}
            for r in self._call_log:
                if r.use_case not in by_use_case:
                    by_use_case[r.use_case] = {"calls": 0, "cost": 0.0, "tokens": 0}
                by_use_case[r.use_case]["calls"]  += 1
                by_use_case[r.use_case]["cost"]   += r.cost_usd
                by_use_case[r.use_case]["tokens"] += r.input_tokens + r.output_tokens

            return CostReport(
                total_calls=total_calls,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                avg_latency_ms=avg_latency,
                by_use_case=by_use_case
            )


if __name__ == "__main__":
    load_dotenv()

    # --- Test 1: normal classification + extraction ---
    client = EnterpriseLLMClient()
    filings = load_all_filings()[:3]  # limit to 3 to save tokens

    for filing in filings:
        # classification call
        client.complete(
            messages=[
                {"role": "system", "content": "Classify this SEC filing. Return JSON with filing_type and company_name."},
                {"role": "user",   "content": filing["content"][:2000]}  # truncate for speed
            ],
            use_case="classification"
        )

        # extraction call
        client.complete(
            messages=[
                {"role": "system", "content": "Extract the company name and CIK number. Return JSON with company_name and cik."},
                {"role": "user",   "content": filing["content"][:2000]}
            ],
            use_case="extraction"
        )

    print("\n=== Cost Report ===")
    report = client.get_cost_report()
    print(f"Total Calls:    {report.total_calls}")
    print(f"Total Tokens:   {report.total_tokens}")
    print(f"Total Cost:     ${report.total_cost_usd:.6f}")
    print(f"Avg Latency:    {report.avg_latency_ms:.0f}ms")
    print(f"\nBy Use Case:")
    for use_case, stats in report.by_use_case.items():
        print(f"  {use_case}: {stats['calls']} calls | {stats['tokens']} tokens | ${stats['cost']:.6f}")

    # --- Test 2: budget enforcement ---
    print("\n=== Budget Enforcement Test ===")
    tiny_budget_client = EnterpriseLLMClient(budget_limit_usd=0.0001)
    try:
        for i in range(3):
            tiny_budget_client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond in JSON with key 'answer'."},
                    {"role": "user",   "content": "What is the SEC?"}
                ],
                use_case="budget_test"
            )
            print(f"Call {i+1} succeeded")
    except BudgetExceededError as e:
        print(f"BudgetExceededError caught as expected: {e}")