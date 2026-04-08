"""
ChatPromptTemplate and Variable Injection

Demonstrates prompt templates with:
- System/human/AI message types
- Variable interpolation
- Partial templates
- Few-shot message patterns in templates

Usage:
    python 04-prompt-templates.py

Requires:
    - Azure OpenAI endpoint configured
    - langchain, langchain-openai packages
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0,
)


def basic_template_demo() -> None:
    """Demonstrate basic variable interpolation in prompt templates."""
    print("=" * 60)
    print("BASIC TEMPLATE: Variable Injection")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SEC filing analyst specializing in {domain}. "
                   "Provide analysis in {format} format."),
        ("human", "Analyze this excerpt from a {filing_type} filing:\n\n{document}"),
    ])

    chain = prompt | model | StrOutputParser()

    result = chain.invoke({
        "domain": "risk factor disclosure",
        "format": "bullet-point",
        "filing_type": "10-K",
        "document": (
            "ITEM 1A. RISK FACTORS: The Company faces significant cybersecurity "
            "threats. In 2024, we detected 847 attempted intrusions, a 34% increase "
            "from 2023. We invested $120M in security infrastructure. The Company's "
            "operations across 12 jurisdictions expose it to regulatory complexity."
        ),
    })

    print(f"\nResult:\n{result}")

    formatted = prompt.invoke({
        "domain": "risk factor disclosure",
        "format": "bullet-point",
        "filing_type": "10-K",
        "document": "...(document text)...",
    })
    print(f"\nFormatted prompt messages:")
    for msg in formatted.messages:
        print(f"  [{msg.type}] {msg.content[:80]}...")


def partial_template_demo() -> None:
    """Demonstrate partial templates — fixing some variables at setup time."""
    print(f"\n{'=' * 60}")
    print("PARTIAL TEMPLATE: Pre-Filled Variables")
    print("=" * 60)

    generic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {document_domain} classifier. "
                   "Classify documents into exactly one of: {categories}. "
                   "Return only the category name, nothing else."),
        ("human", "{document}"),
    ])

    sec_classifier = generic_prompt.partial(
        document_domain="SEC filing",
        categories="10-K, 10-Q, 8-K, DEF 14A, S-1",
    )

    incident_classifier = generic_prompt.partial(
        document_domain="IT incident",
        categories="P1-Critical, P2-High, P3-Medium, P4-Low",
    )

    chain_sec = sec_classifier | model | StrOutputParser()
    chain_incident = incident_classifier | model | StrOutputParser()

    sec_result = chain_sec.invoke({
        "document": "Annual report for fiscal year ended December 31, 2024. "
                    "Item 7: Management's Discussion and Analysis."
    })
    print(f"\n  SEC classification: {sec_result.strip()}")

    incident_result = chain_incident.invoke({
        "document": "Production database failover triggered at 02:30 UTC. "
                    "Customer-facing API returning 503 errors. "
                    "Estimated 40% of requests affected across all regions."
    })
    print(f"  Incident classification: {incident_result.strip()}")


def few_shot_template_demo() -> None:
    """Demonstrate few-shot examples in a prompt template."""
    print(f"\n{'=' * 60}")
    print("FEW-SHOT TEMPLATE")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a contract clause classifier. Classify each clause into "
         "one of: indemnification, limitation_of_liability, termination, "
         "confidentiality, intellectual_property, force_majeure, other. "
         "Return only the classification."),
        ("human", "Classify: 'Neither party shall be liable for any indirect, "
                  "consequential, or punitive damages arising from this agreement.'"),
        ("ai", "limitation_of_liability"),
        ("human", "Classify: 'Either party may terminate this agreement upon 90 days "
                  "written notice, or immediately upon material breach.'"),
        ("ai", "termination"),
        ("human", "Classify: 'The receiving party shall maintain the confidentiality of "
                  "all proprietary information for a period of 5 years following disclosure.'"),
        ("ai", "confidentiality"),
        ("human", "Classify: '{clause}'"),
    ])

    chain = prompt | model | StrOutputParser()

    test_clauses = [
        "Provider shall defend and hold harmless Client from all claims arising "
        "from Provider's negligence or willful misconduct.",

        "All inventions, developments, and improvements created during the "
        "engagement shall be the sole property of the Client.",

        "Neither party shall be deemed in default if performance is delayed "
        "by acts of God, government restrictions, or pandemic.",
    ]

    for clause in test_clauses:
        result = chain.invoke({"clause": clause})
        print(f"\n  Clause: {clause[:70]}...")
        print(f"  Classification: {result.strip()}")


def main() -> None:
    basic_template_demo()
    partial_template_demo()
    few_shot_template_demo()


if __name__ == "__main__":
    main()
