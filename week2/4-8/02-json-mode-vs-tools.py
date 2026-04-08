"""
JSON Mode vs. Function Calling

Demonstrates the difference between:
- JSON mode: constraining the LLM to produce valid JSON (for extraction)
- Function calling: enabling the LLM to invoke external tools (for actions)

Both are API-level constraints, not prompt-level instructions.

Usage:
    python 02-json-mode-vs-tools.py

Requires:
    - Azure OpenAI endpoint configured
"""

import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)


def extract_with_json_mode() -> None:
    """Use JSON mode to force valid JSON output for entity extraction."""
    print("=" * 60)
    print("JSON MODE: Invoice Entity Extraction")
    print("=" * 60)

    invoice_text = """
    INVOICE #INV-2024-7734
    From: Horizon Cloud Services, LLC
    Date: November 20, 2024
    Due: December 20, 2024

    Description                     Qty    Rate        Amount
    Cloud Infrastructure (monthly)    1    $42,500.00  $42,500.00
    Premium Support SLA               1    $8,750.00   $8,750.00
    Data Egress Overage (TB)         12    $90.00      $1,080.00

    Subtotal: $52,330.00
    Tax (8.25%): $4,317.23
    TOTAL DUE: $56,647.23

    Payment Terms: Net 30
    Wire to: Chase Bank, Acct #xxxx-4892
    """

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured data from invoices. Return a JSON object with: "
                    "vendor_name, invoice_number, invoice_date (YYYY-MM-DD), "
                    "line_items (array of {description, quantity, unit_price, total}), "
                    "subtotal, tax, total, payment_terms."
                ),
            },
            {"role": "user", "content": invoice_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    content = response.choices[0].message.content or ""
    data = json.loads(content)

    print(f"\nJSON Mode guarantees valid JSON: {isinstance(data, dict)}")
    print(f"Vendor: {data.get('vendor_name')}")
    print(f"Invoice #: {data.get('invoice_number')}")
    print(f"Line items: {len(data.get('line_items', []))}")
    print(f"Total: {data.get('total')}")
    print(f"\nFull extraction:\n{json.dumps(data, indent=2)}")


def search_vendor_registry(vendor_name: str) -> str:
    """Simulated vendor registry lookup."""
    registry = {
        "horizon cloud services": {
            "vendor_id": "VND-2847",
            "status": "approved",
            "risk_tier": "tier_2",
            "contract_end": "2025-06-30",
            "payment_method": "wire",
        },
    }
    key = vendor_name.lower().replace(", llc", "").replace(" llc", "").strip()
    if key in registry:
        return json.dumps(registry[key], indent=2)
    return json.dumps({"error": f"Vendor '{vendor_name}' not found in registry"})


def validate_invoice_against_contract(
    vendor_id: str, invoice_total: float, period: str
) -> str:
    """Simulated contract validation."""
    contracts = {
        "VND-2847": {
            "monthly_cap": 55000.00,
            "approved_services": [
                "Cloud Infrastructure",
                "Premium Support SLA",
                "Data Egress",
            ],
        },
    }
    contract = contracts.get(vendor_id)
    if not contract:
        return json.dumps({"error": f"No contract found for {vendor_id}"})

    issues = []
    if invoice_total > contract["monthly_cap"]:
        issues.append(
            f"Invoice total ${invoice_total:,.2f} exceeds monthly cap "
            f"${contract['monthly_cap']:,.2f}"
        )

    return json.dumps({
        "vendor_id": vendor_id,
        "monthly_cap": contract["monthly_cap"],
        "invoice_total": invoice_total,
        "within_cap": invoice_total <= contract["monthly_cap"],
        "issues": issues,
    }, indent=2)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_vendor_registry",
            "description": "Look up a vendor in the approved vendor registry by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_name": {"type": "string", "description": "The vendor name to search for."},
                },
                "required": ["vendor_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_invoice_against_contract",
            "description": "Validate an invoice total against the vendor's contract terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_id": {"type": "string", "description": "Vendor ID from the registry."},
                    "invoice_total": {"type": "number", "description": "The invoice total amount."},
                    "period": {"type": "string", "description": "The billing period (e.g., '2024-11')."},
                },
                "required": ["vendor_id", "invoice_total", "period"],
            },
        },
    },
]

TOOL_MAP = {
    "search_vendor_registry": search_vendor_registry,
    "validate_invoice_against_contract": validate_invoice_against_contract,
}


def process_with_function_calling() -> None:
    """Use function calling to validate an invoice against enterprise systems."""
    print(f"\n{'=' * 60}")
    print("FUNCTION CALLING: Invoice Validation Workflow")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an accounts payable assistant. When given an invoice, "
                "look up the vendor in the registry and validate the total against "
                "the contract. Report any issues."
            ),
        },
        {
            "role": "user",
            "content": (
                "Process this invoice: Invoice #INV-2024-7734 from Horizon Cloud Services, LLC. "
                "Total: $56,647.23. Period: November 2024."
            ),
        },
    ]

    max_iterations = 5
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )

        message = response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            print(f"\n  Final response (after {iteration} tool calls):")
            print(f"  {message.content}")
            break

        for tc in message.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"\n  Tool call: {func_name}({json.dumps(func_args)})")

            result = TOOL_MAP[func_name](**func_args)
            print(f"  Result: {result[:120]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


def main() -> None:
    extract_with_json_mode()
    process_with_function_calling()

    print(f"\n{'=' * 60}")
    print("KEY DIFFERENCES")
    print("=" * 60)
    print("  JSON Mode:")
    print("    - Forces ALL output to be valid JSON")
    print("    - Used for extraction / structured output")
    print("    - No external function execution")
    print("    - One API call, one JSON response")
    print()
    print("  Function Calling:")
    print("    - LLM decides WHETHER to call a tool")
    print("    - Used for interacting with external systems")
    print("    - Your code executes the function")
    print("    - Multiple API calls in a loop")


if __name__ == "__main__":
    main()
