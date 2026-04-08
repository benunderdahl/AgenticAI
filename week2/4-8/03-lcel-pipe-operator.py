"""
LCEL Pipe Operator and Runnable Composition

Demonstrates the LangChain Expression Language (LCEL):
- The pipe operator (|) for composing Runnables
- RunnablePassthrough for preserving context
- RunnableLambda for custom transformations
- invoke(), batch(), and stream() on chains

Usage:
    python 03-lcel-pipe-operator.py

Requires:
    - Azure OpenAI endpoint configured
    - langchain, langchain-openai packages
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_CONFIG = {
    "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    "temperature": 0,
}


def basic_chain_demo() -> None:
    """The simplest LCEL chain: prompt -> model -> string output."""
    print("=" * 60)
    print("BASIC CHAIN: Prompt | Model | Parser")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst. Provide concise assessments."),
        ("human", "Assess the risk level of this event: {event}"),
    ])

    model = AzureChatOpenAI(**AZURE_CONFIG)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({
        "event": "A Fortune 500 company disclosed a material weakness in "
                 "internal controls over revenue recognition in their 10-K."
    })
    print(f"\nResult:\n{result}")


def batch_demo() -> None:
    """Process multiple inputs in parallel with .batch()."""
    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the SEC filing type in one word: 10-K, 10-Q, 8-K, DEF-14A, or S-1."),
        ("human", "{excerpt}"),
    ])

    model = AzureChatOpenAI(**AZURE_CONFIG)
    chain = prompt | model | StrOutputParser()

    excerpts = [
        {"excerpt": "Annual report for fiscal year ended December 31, 2024. Item 1A Risk Factors."},
        {"excerpt": "Quarterly report for period ended March 31, 2024. Q1 financial statements."},
        {"excerpt": "Current report: the Company entered into a merger agreement on October 15."},
        {"excerpt": "Definitive proxy statement. Notice of annual meeting. Election of directors."},
    ]

    results = chain.batch(excerpts, config={"max_concurrency": 3})

    for excerpt, result in zip(excerpts, results):
        print(f"  {excerpt['excerpt'][:60]}... -> {result.strip()}")


def streaming_demo() -> None:
    """Stream tokens as they are generated."""
    print(f"\n{'=' * 60}")
    print("STREAMING")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a compliance analyst."),
        ("human", "Briefly explain why SOX Section 302 certifications matter for {filing_type} filings."),
    ])

    model = AzureChatOpenAI(**AZURE_CONFIG)
    chain = prompt | model | StrOutputParser()

    print("\n  Streaming response: ", end="")
    for chunk in chain.stream({"filing_type": "10-K"}):
        print(chunk, end="", flush=True)
    print()


def passthrough_and_lambda_demo() -> None:
    """Use RunnablePassthrough and RunnableLambda for data transformation."""
    print(f"\n{'=' * 60}")
    print("RUNNABLEPASSTHROUGH AND RUNNABLELAMBDA")
    print("=" * 60)

    def preprocess_document(input_dict: dict) -> dict:
        """Add metadata and truncate long documents."""
        text = input_dict["document"]
        return {
            "document": text[:2000],
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def format_assessment(ai_response: str) -> dict:
        """Post-process the LLM output into a structured result."""
        return {
            "assessment": ai_response.strip(),
            "word_count": len(ai_response.split()),
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Assess this document for regulatory compliance risks. Be concise."),
        ("human", "Document ({char_count} chars, {word_count} words):\n\n{document}"),
    ])

    model = AzureChatOpenAI(**AZURE_CONFIG)

    chain = (
        RunnableLambda(preprocess_document)
        | prompt
        | model
        | StrOutputParser()
        | RunnableLambda(format_assessment)
    )

    result = chain.invoke({
        "document": (
            "ITEM 1A RISK FACTORS: The Company's operations in 12 jurisdictions "
            "expose it to diverse regulatory requirements. Recent changes to EU "
            "data protection law (GDPR Article 83) could result in fines up to "
            "4% of global annual revenue. The Company has not yet completed its "
            "impact assessment for the EU AI Act, which takes effect in 2025."
        )
    })

    print(f"\n  Assessment: {result['assessment'][:200]}...")
    print(f"  Response words: {result['word_count']}")

    print(f"\n  --- RunnablePassthrough ---")

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this risk factor in exactly one sentence."),
        ("human", "{text}"),
    ])

    model_mini = AzureChatOpenAI(**AZURE_CONFIG)

    chain_with_passthrough = RunnablePassthrough.assign(
        summary=summarize_prompt | model_mini | StrOutputParser()
    )

    result = chain_with_passthrough.invoke({
        "text": "The Company faces cybersecurity threats from state-sponsored actors "
                "targeting our defense industry clients. In 2024, we detected 847 "
                "intrusion attempts, a 34% increase from the prior year."
    })

    print(f"  Original text: {result['text'][:80]}...")
    print(f"  Summary: {result['summary']}")


def main() -> None:
    basic_chain_demo()
    batch_demo()
    streaming_demo()
    passthrough_and_lambda_demo()


if __name__ == "__main__":
    main()
