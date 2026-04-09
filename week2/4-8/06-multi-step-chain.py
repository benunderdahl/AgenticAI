"""
Example 06 — Multi-Step Chain: Classify -> Extract -> Assess

Composes multiple LCEL chains into a pipeline that:
1. Classifies a document
2. Extracts entities (using classification context)
3. Produces a risk assessment

Demonstrates RunnableLambda for bridging chain outputs to inputs.

Usage:
    python 06-multi-step-chain.py

Requires:
    - Azure OpenAI endpoint configured
    - langchain, langchain-openai packages
"""

import time
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview",
    temperature=0,
)

class DocumentClassification(BaseModel):
    document_type: str = Field(
        description="One of: risk_assessment, compliance_report, incident_report, "
        "financial_filing, contract, policy_document"
    )
    confidence: float = Field(description="0.0 to 1.0")
    key_topics: list[str] = Field(description="Main topics identified")


classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify this document. Identify its type and key topics."),
    ("human", "{document}"),
])

classify_chain = classify_prompt | model.with_structured_output(DocumentClassification)


class DocumentEntities(BaseModel):
    organizations: list[str] = Field(description="Organization names")
    monetary_amounts: list[str] = Field(description="Dollar amounts with context")
    dates: list[str] = Field(description="Significant dates")
    regulations: list[str] = Field(description="Regulatory references")
    people: list[str] = Field(description="Named individuals and their roles")


extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract all entities from this {document_type} document. "
     "Focus on: organizations, monetary amounts, dates, regulations, and people."),
    ("human", "{document}"),
])

extract_chain = extract_prompt | model.with_structured_output(DocumentEntities)


class RiskAssessment(BaseModel):
    risk_score: float = Field(description="Overall risk 0.0 to 1.0")
    risk_factors: list[str] = Field(description="Identified risks, ordered by severity")
    recommended_actions: list[str] = Field(description="Recommended next steps")
    summary: str = Field(description="One-paragraph risk summary")


assess_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Assess the risk level of this document. Consider: regulatory exposure, "
     "financial impact, operational risk, and reputational risk."),
    ("human",
     "Document type: {document_type}\n"
     "Key topics: {topics}\n"
     "Organizations: {organizations}\n"
     "Monetary amounts: {amounts}\n"
     "Regulations: {regulations}\n\n"
     "Original document:\n{document}"),
])

assess_chain = assess_prompt | model.with_structured_output(RiskAssessment)


_original_document = ""


def step1_classify(input_dict: dict) -> dict:
    """Run classification and pass results forward."""
    global _original_document
    _original_document = input_dict["document"]

    classification = classify_chain.invoke(input_dict)
    return {
        "document": _original_document,
        "document_type": classification.document_type,
        "confidence": classification.confidence,
        "topics": classification.key_topics,
    }


def step2_extract(prev: dict) -> dict:
    """Run extraction using classification context."""
    entities = extract_chain.invoke({
        "document": prev["document"],
        "document_type": prev["document_type"],
    })
    return {
        **prev,
        "organizations": entities.organizations,
        "amounts": entities.monetary_amounts,
        "dates": entities.dates,
        "regulations": entities.regulations,
        "people": entities.people,
    }


def step3_assess(prev: dict) -> dict:
    """Run risk assessment using all gathered context."""
    assessment = assess_chain.invoke({
        "document": prev["document"],
        "document_type": prev["document_type"],
        "topics": ", ".join(prev["topics"]),
        "organizations": ", ".join(prev["organizations"]),
        "amounts": ", ".join(prev["amounts"]),
        "regulations": ", ".join(prev["regulations"]),
    })
    return {
        "classification": {
            "type": prev["document_type"],
            "confidence": prev["confidence"],
            "topics": prev["topics"],
        },
        "entities": {
            "organizations": prev["organizations"],
            "amounts": prev["amounts"],
            "dates": prev["dates"],
            "regulations": prev["regulations"],
            "people": prev["people"],
        },
        "assessment": {
            "risk_score": assessment.risk_score,
            "risk_factors": assessment.risk_factors,
            "recommended_actions": assessment.recommended_actions,
            "summary": assessment.summary,
        },
    }


pipeline = (
    RunnableLambda(step1_classify)
    | RunnableLambda(step2_extract)
    | RunnableLambda(step3_assess)
)


SAMPLE_DOCUMENT = """
COMPLIANCE AUDIT REPORT — Q3 2024

Client: Apex Financial Services, Inc.
Audit Period: July 1 — September 30, 2024
Audit Firm: EY Advisory Services

EXECUTIVE SUMMARY:
The audit identified 7 findings across BSA/AML, SOX, and GDPR domains.
Two findings are classified as critical: (1) OFAC screening gaps in the
correspondent banking channel affecting ~$180M in monthly wire volume,
and (2) a SOX-significant deficiency in the segregation of duties for
the general ledger close process.

FINDING 1 (CRITICAL): BSA/AML — OFAC Screening Gap
Wire transfers processed through the correspondent banking network bypass
the ComplianceGuard OFAC screening module. Approximately 12,000 transactions
per month (~$180M) are affected. The gap was introduced during the
ComplianceGuard v5.0 upgrade in June 2024. Estimated remediation cost: $280,000.
Target completion: December 15, 2024.

FINDING 2 (CRITICAL): SOX — Segregation of Duties
Three individuals in the Finance team have both journal entry creation and
approval access in the ERP system (SAP S/4HANA). This creates a risk of
unauthorized entries. The issue was flagged by PwC during the interim audit
review. Remediation requires CyberArk role restructuring. Cost: $45,000.

FINDING 3 (HIGH): GDPR — Cross-Border Data Transfer
Customer PII from EU operations is replicated to a US-based analytics
platform without Standard Contractual Clauses (SCCs). Affects approximately
2.3 million EU data subject records. DPO has initiated SCC negotiation
with the analytics vendor (DataInsight Corp). Target: November 30, 2024.
"""

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-STEP PIPELINE: Classify -> Extract -> Assess")
    print("=" * 60)

    start = time.perf_counter()
    result = pipeline.invoke({"document": SAMPLE_DOCUMENT})
    elapsed = time.perf_counter() - start

    print(f"\n  CLASSIFICATION:")
    print(f"    Type: {result['classification']['type']}")
    print(f"    Confidence: {result['classification']['confidence']:.0%}")
    print(f"    Topics: {', '.join(result['classification']['topics'])}")

    print(f"\n  ENTITIES:")
    for key, values in result["entities"].items():
        if values:
            print(f"    {key}: {', '.join(values[:5])}")

    print(f"\n  RISK ASSESSMENT:")
    print(f"    Score: {result['assessment']['risk_score']}")
    print(f"    Summary: {result['assessment']['summary'][:200]}")
    print(f"    Risk factors:")
    for rf in result["assessment"]["risk_factors"]:
        print(f"      - {rf}")
    print(f"    Recommended actions:")
    for action in result["assessment"]["recommended_actions"]:
        print(f"      - {action}")

    print(f"\n  Pipeline time: {elapsed:.1f}s")
