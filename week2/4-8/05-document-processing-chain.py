"""
Document Processing Chain with Structured Output

Builds a typed chain that classifies and extracts entities from
SEC filings using ChatPromptTemplate + with_structured_output().
Replaces manual JSON parsing with Pydantic-validated output.

Usage:
    python 05-document-processing-chain.py

Requires:
    - Azure OpenAI endpoint configured
    - langchain, langchain-openai packages
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import os
load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview",
    temperature=0,
)


class FilingClassification(BaseModel):
    """Classification result for an SEC filing."""

    filing_type: str = Field(
        description="The SEC form type: '10-K', '10-Q', '8-K', 'DEF 14A', 'S-1', or 'UNKNOWN'"
    )
    confidence: float = Field(
        description="Classification confidence from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )
    key_signals: list[str] = Field(
        description="Specific textual signals that support the classification"
    )


class ExtractedEntities(BaseModel):
    """Entities extracted from a financial document."""

    company_name: str = Field(description="The primary company in the filing")
    filing_period: str = Field(
        description="The reporting period (e.g., 'FY 2024', 'Q3 2024')"
    )
    revenue: str | None = Field(
        default=None,
        description="Total revenue mentioned, with currency (e.g., '$3.2B')"
    )
    key_events: list[str] = Field(
        description="Material events, decisions, or changes disclosed"
    )
    regulations_cited: list[str] = Field(
        description="Regulations, standards, or laws mentioned"
    )
    risk_factors: list[str] = Field(
        description="Risk factors or concerns identified"
    )


def classification_chain_demo() -> None:
    """Build and run a typed classification chain."""
    print("=" * 60)
    print("CLASSIFICATION CHAIN (Structured Output)")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the provided SEC filing excerpt into its form type. "
         "Identify the specific textual signals that support your classification."),
        ("human", "{document}"),
    ])

    chain = prompt | model.with_structured_output(FilingClassification)

    test_documents = [
        {
            "label": "Annual report",
            "text": (
                "FORM 10-K. Annual Report for the fiscal year ended December 31, 2024. "
                "Quantum Analytics Corp. Item 1: Business Overview. Item 1A: Risk Factors. "
                "Item 7: Management's Discussion and Analysis. Total revenue: $8.7 billion."
            ),
        },
        {
            "label": "Material event",
            "text": (
                "CURRENT REPORT ON FORM 8-K. Date of Report: November 15, 2024. "
                "NexGen Robotics Inc. Item 1.01: Entry into a Material Definitive Agreement. "
                "The Company entered into a merger agreement to acquire DataVault Corp "
                "for $2.1 billion in cash and stock."
            ),
        },
        {
            "label": "Ambiguous excerpt",
            "text": (
                "Securities and Exchange Commission. The Company reported total revenues "
                "and operating expenses for the reporting period. Management discusses "
                "trends in the business. Risk factors are summarized below."
            ),
        },
    ]

    for doc in test_documents:
        print(f"\n  --- {doc['label']} ---")
        result: FilingClassification = chain.invoke({"document": doc["text"]})
        print(f"  Type: {result.filing_type} (confidence: {result.confidence:.0%})")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Signals: {result.key_signals}")


def extraction_chain_demo() -> None:
    """Build and run a typed entity extraction chain."""
    print(f"\n{'=' * 60}")
    print("EXTRACTION CHAIN (Structured Output)")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract all key entities from this SEC filing excerpt. "
         "Be thorough — capture every company name, monetary amount, "
         "regulatory reference, and risk factor mentioned."),
        ("human", "{document}"),
    ])

    chain = prompt | model.with_structured_output(ExtractedEntities)

    filing_text = (
        "FORM 8-K — CURRENT REPORT. Meridian Financial Group, Inc. "
        "Date of Report: January 15, 2025. "
        "Item 1.01: Entry into a Material Definitive Agreement. "
        "The Company entered into a definitive agreement to acquire "
        "Vertex Analytics LLC for $340 million in cash. The transaction "
        "is subject to Hart-Scott-Rodino Act review and approval by "
        "the Federal Trade Commission. Expected to close Q3 2025. "
        "The acquisition will be funded through existing cash reserves "
        "($180M) and a new $200M credit facility with JPMorgan Chase. "
        "Item 5.02: Departure of Directors or Certain Officers. "
        "CFO James Wilson resigned effective January 31, 2025. "
        "The Board appointed interim CFO Lisa Okafor."
    )

    result: ExtractedEntities = chain.invoke({"document": filing_text})

    print(f"\n  Company: {result.company_name}")
    print(f"  Period: {result.filing_period}")
    print(f"  Revenue: {result.revenue}")
    print(f"  Key events ({len(result.key_events)}):")
    for event in result.key_events:
        print(f"    - {event}")
    print(f"  Regulations ({len(result.regulations_cited)}):")
    for reg in result.regulations_cited:
        print(f"    - {reg}")
    print(f"  Risk factors ({len(result.risk_factors)}):")
    for risk in result.risk_factors:
        print(f"    - {risk}")


def batch_demo() -> None:
    """Process multiple documents in parallel with .batch()."""
    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the SEC filing. Be concise."),
        ("human", "{document}"),
    ])

    chain = prompt | model.with_structured_output(FilingClassification)

    documents = [
        {"document": "Annual report, fiscal year ended December 31, 2024. Item 7 MD&A."},
        {"document": "Quarterly report, period ended September 30, 2024. Q3 financials."},
        {"document": "Definitive proxy statement. Annual meeting. Director elections."},
        {"document": "Registration statement under Securities Act of 1933. IPO offering."},
    ]

    results = chain.batch(documents, config={"max_concurrency": 3})

    for doc, result in zip(documents, results):
        print(f"  {doc['document'][:50]}... -> {result.filing_type} ({result.confidence:.0%})")


def main() -> None:
    classification_chain_demo()
    extraction_chain_demo()
    batch_demo()


if __name__ == "__main__":
    main()
