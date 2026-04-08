"""
Function Calling Basics (Raw OpenAI API)

Demonstrates the complete function calling lifecycle:
1. Register tool schemas with the API
2. LLM decides to call a tool and returns structured arguments
3. Your code executes the tool
4. Result is sent back to the LLM for final response

Usage:
    python 01-function-calling-basics.py

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_compliance_database",
            "description": (
                "Search the compliance findings database for issues matching "
                "the given regulation and severity filter. Returns a list of "
                "open and in-remediation findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "Regulation code to filter by (e.g., 'SOX', 'GDPR', 'PCI-DSS', 'BSA/AML').",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "all"],
                        "description": "Severity level filter. Use 'all' for no filter.",
                    },
                },
                "required": ["regulation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_remediation_timeline",
            "description": (
                "Get the detailed remediation plan and timeline for a specific "
                "compliance finding by its ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finding_id": {
                        "type": "string",
                        "description": "The finding identifier (e.g., 'SOX-2024-033').",
                    },
                },
                "required": ["finding_id"],
            },
        },
    },
]


def search_compliance_database(regulation: str, severity: str = "all") -> str:
    """Simulated compliance database search."""
    findings = [
        {"id": "SOX-2024-033", "regulation": "SOX", "severity": "high",
         "title": "Stale privileged accounts on RegReportDB",
         "status": "open", "days_open": 45},
        {"id": "SOX-2024-028", "regulation": "SOX", "severity": "medium",
         "title": "Change management documentation gap",
         "status": "in_remediation", "days_open": 82},
        {"id": "GDPR-2024-018", "regulation": "GDPR", "severity": "high",
         "title": "Cross-border PII transfer without SCCs",
         "status": "in_remediation", "days_open": 63},
        {"id": "PCI-2024-007", "regulation": "PCI-DSS", "severity": "medium",
         "title": "Outdated incident response runbook",
         "status": "open", "days_open": 104},
        {"id": "BSA-2024-041", "regulation": "BSA/AML", "severity": "critical",
         "title": "OFAC screening gap in correspondent banking",
         "status": "open", "days_open": 38},
    ]

    results = [
        f for f in findings
        if (regulation.upper() in f["regulation"].upper() or regulation.lower() == "all")
        and (severity == "all" or f["severity"] == severity)
    ]
    return json.dumps({"count": len(results), "findings": results}, indent=2)


def get_remediation_timeline(finding_id: str) -> str:
    """Simulated remediation timeline lookup."""
    timelines = {
        "SOX-2024-033": {
            "finding_id": "SOX-2024-033",
            "assigned_team": "IAM Operations",
            "target_date": "2024-11-15",
            "milestones": [
                {"date": "2024-10-01", "milestone": "Access review initiated", "status": "complete"},
                {"date": "2024-10-15", "milestone": "Stale accounts identified", "status": "complete"},
                {"date": "2024-11-01", "milestone": "CyberArk recertification campaign", "status": "in_progress"},
                {"date": "2024-11-15", "milestone": "Full remediation and validation", "status": "pending"},
            ],
            "blockers": ["CyberArk v12.5 upgrade required for certification campaign feature"],
            "estimated_cost": "$45,000",
        },
        "BSA-2024-041": {
            "finding_id": "BSA-2024-041",
            "assigned_team": "Transaction Monitoring Engineering",
            "target_date": "2024-12-15",
            "milestones": [
                {"date": "2024-09-20", "milestone": "Gap analysis complete", "status": "complete"},
                {"date": "2024-10-15", "milestone": "Pipeline redesign specification", "status": "in_progress"},
                {"date": "2024-11-15", "milestone": "Testing environment provisioning", "status": "pending"},
                {"date": "2024-12-15", "milestone": "Production deployment and validation", "status": "pending"},
            ],
            "blockers": ["Waiting on ComplianceGuard v5.3 SDK release", "Testing env provisioning delayed"],
            "estimated_cost": "$280,000",
        },
    }
    if finding_id in timelines:
        return json.dumps(timelines[finding_id], indent=2)
    return json.dumps({"error": f"Finding {finding_id} not found in remediation tracker"})


TOOL_MAP = {
    "search_compliance_database": search_compliance_database,
    "get_remediation_timeline": get_remediation_timeline,
}


def run_function_calling_demo() -> None:
    """Demonstrate the complete function calling lifecycle."""
    print("=" * 60)
    print("STEP 1: Send message with tool schemas registered")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a compliance operations assistant. Use the available tools "
                "to look up findings and remediation timelines. Always cite finding IDs."
            ),
        },
        {
            "role": "user",
            "content": (
                "What SOX compliance findings do we have open? "
                "For any high-severity ones, get the remediation timeline."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )

    assistant_message = response.choices[0].message
    print(f"\nFinish reason: {response.choices[0].finish_reason}")
    print(f"Tool calls requested: {len(assistant_message.tool_calls or [])}")

    print(f"\n{'=' * 60}")
    print("STEP 2: Execute requested tool calls")
    print("=" * 60)

    messages.append(assistant_message.model_dump())

    for tool_call in assistant_message.tool_calls or []:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"\n  Calling: {func_name}({func_args})")

        if func_name in TOOL_MAP:
            result = TOOL_MAP[func_name](**func_args)
        else:
            result = json.dumps({"error": f"Unknown function: {func_name}"})

        print(f"  Result preview: {result[:150]}...")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    print(f"\n{'=' * 60}")
    print("STEP 3: LLM processes tool results")
    print("=" * 60)

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )

    next_message = response.choices[0].message

    if next_message.tool_calls:
        print(f"\nModel wants more tool calls: {len(next_message.tool_calls)}")
        messages.append(next_message.model_dump())

        for tool_call in next_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  Calling: {func_name}({func_args})")

            result = TOOL_MAP.get(func_name, lambda **kw: '{"error": "unknown"}')(**func_args)
            print(f"  Result preview: {result[:150]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        next_message = response.choices[0].message

    print(f"\n{'=' * 60}")
    print("STEP 4: Final response")
    print("=" * 60)
    print(next_message.content)

    print(f"\n{'=' * 60}")
    print("MESSAGE TRACE")
    print("=" * 60)
    for i, msg in enumerate(messages):
        role = msg.get("role", msg.role if hasattr(msg, "role") else "?")
        if role == "tool":
            print(f"  [{i}] tool (call_id: {msg.get('tool_call_id', '?')[:20]}...)")
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = [tc.function.name for tc in msg.tool_calls]
            print(f"  [{i}] assistant [tool_calls: {calls}]")
        else:
            content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
            print(f"  [{i}] {role}: {content[:80]}...")


if __name__ == "__main__":
    run_function_calling_demo()
