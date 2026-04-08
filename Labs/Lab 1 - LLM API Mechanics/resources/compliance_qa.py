from pydantic import BaseModel, ValidationError
import tiktoken
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from json import JSONDecodeError
from filing_classifier import PRICING

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

class ComplianceQASession:
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        max_context_tokens: int = 12_000,  # Intentionally low to trigger windowing
        max_output_tokens: int = 500,
    ):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
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
                    "You MUST respond in JSON format "
                ),
            }
        ]

    def ask(self, question: str) -> QAResponse:
        self.messages.append({"role": "user", "content": question})

        context_was_trimmed = False
        token_limit = self.max_context_tokens - self.max_output_tokens

        while self._count_messages_tokens() > token_limit:
            if len(self.messages) <= 2:
                break 
                
            before = self._count_messages_tokens()
            removed = self.messages[1:3]
            self.messages = [self.messages[0]] + self.messages[3:]
            after = self._count_messages_tokens()
            freed = before - after 
            print(f"  [TRIM] Removed {len(removed)} messages, freed {freed} tokens")

            self.messages.insert(1, {
                "role": "system",
                "content": "Note: earlier conversation context was trimmed to manage context window size."
            })

            if not context_was_trimmed:
                self.trimming_events += 1
                context_was_trimmed = True
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_format={"type": "json_object"},
                max_tokens=self.max_output_tokens,
                temperature=0.0
            )
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            self.turn_number += 1
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            cost = (
                (input_tokens / 1_000_000) * PRICING['gpt-4o']['input'] +
                (output_tokens / 1_000_000) * PRICING['gpt-4o']['output']
            )
            return QAResponse(
                answer=response.choices[0].message.content,
                turn_number=self.turn_number,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens_in_context=self._count_messages_tokens(),
                context_was_trimmed=context_was_trimmed,
                cost_usd=cost
            )
        except(ValidationError, JSONDecodeError) as e:
            print(e)


    def get_session_stats(self) -> SessionStats:
        total_cost = (
                (self.total_input_tokens / 1_000_000) * PRICING['gpt-4o']["input"] +
                (self.total_output_tokens / 1_000_000) * PRICING['gpt-4o']["output"]
            )
        return SessionStats(
            total_turns=self.turn_number,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost_usd=total_cost,
            trimming_events=self.trimming_events
        )
        

    def _count_messages_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            total += len(self.encoder.encode(msg["content"]))
            total += 4  # OpenAI overhead per message (role + formatting tokens)
        total += 2  # Priming tokens OpenAI adds to every request
        return total
    
if __name__ == "__main__":
    load_dotenv()

    questions = [
    "Give me a comprehensive breakdown of every section required in a 10-K filing under Regulation S-K, including all sub-items and what each must contain.",
    "Expand on each of the risk factor disclosure requirements you mentioned, providing examples of what constitutes adequate vs inadequate disclosure under SEC guidance.",
    "Walk me through the complete executive compensation disclosure requirements under Item 402 of Regulation S-K, including every table that must be included and what data each table requires.",
    "Compare and contrast the compensation disclosure requirements for large accelerated filers vs smaller reporting companies under Item 402, listing every difference in detail.",
    "Give me a detailed explanation of every certification requirement under SOX Sections 302, 404, and 906, including the exact language required and penalties for each.",
    "What are all the internal control reporting requirements under SOX Section 404, including management assessment requirements and auditor attestation requirements?",
    "List every possible SEC enforcement action available against a company that fails to meet 10-K filing deadlines, including civil and criminal penalties with dollar amounts.",
    "How does the SEC's whistleblower program under Dodd-Frank interact with SOX Section 806 protections, and what are the complete procedural requirements for each?",
]
    
    session = ComplianceQASession()
    print("=== Compliance QA Session ===\n")
    print(f"{'Turn':<6} {'Input Tokens':<15} {'Output Tokens':<15} {'Context Tokens':<16} {'Trimmed':<10} {'Cost'}")
    print("-" * 75)

    for q in questions:
        response = session.ask(q)
        if response:
            print(
                f"{response.turn_number:<6} "
                f"{response.input_tokens:<15} "
                f"{response.output_tokens:<15} "
                f"{response.total_tokens_in_context:<16} "
                f"{'YES ⚠️' if response.context_was_trimmed else 'no':<10} "
                f"${response.cost_usd:.6f}"
            )
            print(f"  Q: {q}")
            print(f"  A: {response.answer[:100]}...")
            print()

    print("\n=== Session Summary ===")
    stats = session.get_session_stats()
    print(f"Total Turns:         {stats.total_turns}")
    print(f"Total Input Tokens:  {stats.total_input_tokens}")
    print(f"Total Output Tokens: {stats.total_output_tokens}")
    print(f"Total Cost:          ${stats.total_cost_usd:.6f}")
    print(f"Trimming Events:     {stats.trimming_events}")