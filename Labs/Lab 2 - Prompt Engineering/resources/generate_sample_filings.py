"""
Generates realistic SEC filing excerpts for lab exercises.
Run this script once to populate the sec-filings/ directory.

Usage: python generate_sample_filings.py
"""

import json
import os
import random
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "sec-filings"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Filing templates ---

COMPANIES = [
    ("Meridian Technologies Inc.", "Technology", "0001234567"),
    ("Cascade Financial Group", "Finance", "0002345678"),
    ("BioGenesis Therapeutics", "Healthcare", "0003456789"),
    ("Apex Energy Holdings", "Energy", "0004567890"),
    ("Sentinel Cybersecurity Corp.", "Technology", "0005678901"),
    ("Pacific Reinsurance Ltd.", "Finance", "0006789012"),
    ("NovaPharma International", "Healthcare", "0007890123"),
    ("Continental Logistics Corp.", "Industrials", "0008901234"),
    ("DataVault Cloud Services", "Technology", "0009012345"),
    ("Sterling Wealth Advisors", "Finance", "0001023456"),
    ("GreenPath Renewable Energy", "Energy", "0001134567"),
    ("MedDevice Innovations Inc.", "Healthcare", "0001245678"),
    ("Quantum Computing Solutions", "Technology", "0001356789"),
    ("FirstBridge Capital Partners", "Finance", "0001467890"),
    ("ClearWater Environmental", "Energy", "0001578901"),
]

RISK_FACTORS = [
    "Cybersecurity threats and data breaches could materially impact operations.",
    "Regulatory changes in our primary markets may increase compliance costs.",
    "Supply chain disruptions could affect our ability to deliver products.",
    "Competition from well-capitalized incumbents may erode market share.",
    "Foreign currency fluctuations affect our international revenue streams.",
    "Key personnel departures could disrupt strategic initiatives.",
    "Intellectual property disputes may result in significant legal costs.",
    "Climate-related regulations may require substantial capital investments.",
    "Interest rate volatility affects our borrowing costs and investment returns.",
    "Third-party vendor failures could disrupt critical business processes.",
]

FINANCIAL_ITEMS = [
    "Total revenues of ${revenue}B for the fiscal year ended December 31, 2024.",
    "Net income of ${income}M, representing a {pct}% {direction} from prior year.",
    "Operating cash flow of ${cashflow}M with capital expenditures of ${capex}M.",
    "Total assets of ${assets}B with long-term debt of ${debt}B.",
    "Research and development expenses of ${rnd}M, up {rnd_pct}% year-over-year.",
]


def generate_10k(company: tuple[str, str, str], idx: int) -> str:
    """Generate a realistic 10-K filing excerpt."""
    name, sector, cik = company
    revenue = round(random.uniform(0.5, 25.0), 1)
    income = round(random.uniform(50, 2000))
    employees = random.randint(2000, 100000)

    risk_sample = random.sample(RISK_FACTORS, k=random.randint(3, 6))
    risks = "\n\n".join(f"  - {r}" for r in risk_sample)

    financials = []
    for template in random.sample(FINANCIAL_ITEMS, k=3):
        line = template.replace("${revenue}", str(revenue))
        line = line.replace("${income}", str(income))
        line = line.replace("{pct}", str(random.randint(3, 25)))
        line = line.replace("{direction}", random.choice(["increase", "decrease"]))
        line = line.replace("${cashflow}", str(random.randint(100, 1500)))
        line = line.replace("${capex}", str(random.randint(50, 500)))
        line = line.replace("${assets}", str(round(random.uniform(1, 50), 1)))
        line = line.replace("${debt}", str(round(random.uniform(0.2, 10), 1)))
        line = line.replace("${rnd}", str(random.randint(20, 800)))
        line = line.replace("{rnd_pct}", str(random.randint(5, 30)))
        financials.append(line)

    # Make some filings longer by repeating content
    padding = ""
    if idx % 3 == 0:
        padding = "\n\n" + ("The Company continues to invest in strategic growth "
                           "initiatives across its operating segments. " * 200)

    return f"""UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-K

ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)
OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, 2024
Commission File Number: 001-{random.randint(10000, 99999)}
CIK: {cik}

{name}
Sector: {sector}

ITEM 1. BUSINESS

{name} is a {sector.lower()} company with approximately {employees:,} employees
operating across {random.randint(3, 20)} countries. The Company provides
specialized services and products to enterprise and government customers.

{' '.join(financials)}

ITEM 1A. RISK FACTORS

The following risk factors could materially affect our business, financial
condition, and results of operations:

{risks}

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Management believes the Company is well-positioned for continued growth.
{' '.join(financials)}
{padding}
"""


def generate_8k(company: tuple[str, str, str]) -> str:
    """Generate a realistic 8-K filing excerpt."""
    name, sector, cik = company
    events = [
        (
            "Item 5.02 - Departure of Directors or Certain Officers",
            f"On November 15, 2024, {name} announced that its Chief Financial "
            f"Officer, {random.choice(['Sarah Chen', 'Michael Rodriguez', 'Emily Watson'])}, "
            f"will resign effective December 31, 2024, to pursue other opportunities. "
            f"The Board has appointed interim CFO from the existing leadership team.",
        ),
        (
            "Item 2.01 - Completion of Acquisition or Disposition of Assets",
            f"On October 1, 2024, {name} completed the acquisition of "
            f"{random.choice(['CloudSync Technologies', 'DataPrime Analytics', 'SecureNet Solutions'])} "
            f"for approximately ${random.randint(50, 500)} million in cash and stock. "
            f"The acquisition is expected to be accretive to earnings in fiscal year 2025.",
        ),
        (
            "Item 1.01 - Entry into a Material Definitive Agreement",
            f"{name} entered into a credit agreement providing for a "
            f"${random.uniform(0.5, 5.0):.1f} billion revolving credit facility "
            f"maturing in 2029. The facility replaces the Company's existing "
            f"$2.0 billion facility that was set to mature in 2025.",
        ),
    ]
    event_type, event_text = random.choice(events)

    return f"""UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 8-K

CURRENT REPORT
Pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934

Date of Report: November 15, 2024
Commission File Number: 001-{random.randint(10000, 99999)}
CIK: {cik}

{name}

{event_type}

{event_text}

SIGNATURES

Pursuant to the requirements of the Securities Exchange Act of 1934,
the registrant has duly caused this report to be signed on its behalf
by the undersigned hereunto duly authorized.

{name}
Date: November 15, 2024
"""


def generate_10q(company: tuple[str, str, str]) -> str:
    """Generate a realistic 10-Q filing excerpt."""
    name, sector, cik = company
    quarter = random.choice(["Q1", "Q2", "Q3"])
    revenue = round(random.uniform(0.3, 6.0), 1)

    return f"""UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-Q

QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d)
OF THE SECURITIES EXCHANGE ACT OF 1934

For the quarterly period ended {random.choice(['March 31', 'June 30', 'September 30'])}, 2024
Commission File Number: 001-{random.randint(10000, 99999)}
CIK: {cik}

{name}
Sector: {sector}

PART I - FINANCIAL INFORMATION

Item 1. Financial Statements

{name} reported total revenues of ${revenue}B for {quarter} 2024,
representing a {random.randint(2, 18)}% {random.choice(['increase', 'decrease'])}
compared to the same quarter of the prior year.

Net income for the quarter was ${random.randint(20, 500)}M.
Operating expenses totaled ${round(revenue * random.uniform(0.6, 0.9), 1)}B.

Item 2. Management's Discussion and Analysis

The Company's {quarter} 2024 results reflect continued execution on
strategic priorities including digital transformation, operational
efficiency, and market expansion. Management remains focused on
delivering long-term shareholder value.
"""


def generate_def14a(company: tuple[str, str, str]) -> str:
    """Generate a realistic DEF 14A (proxy statement) excerpt."""
    name, sector, cik = company

    return f"""UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

SCHEDULE 14A
(RULE 14a-101)

INFORMATION REQUIRED IN PROXY STATEMENT

DEFINITIVE PROXY STATEMENT (DEF 14A)

{name}
CIK: {cik}

NOTICE OF ANNUAL MEETING OF STOCKHOLDERS
To Be Held on May 15, 2025

Dear Stockholder:

You are cordially invited to attend the Annual Meeting of Stockholders
of {name} to be held at the Company's headquarters on May 15, 2025.

PROPOSALS TO BE VOTED ON:
1. Election of {random.randint(7, 12)} directors
2. Ratification of {random.choice(['Deloitte & Touche', 'Ernst & Young', 'PricewaterhouseCoopers', 'KPMG'])} LLP
   as independent registered public accounting firm
3. Advisory vote on executive compensation ("Say on Pay")
4. Approval of the 2025 Equity Incentive Plan

EXECUTIVE COMPENSATION SUMMARY:
CEO Total Compensation: ${random.uniform(5, 25):.1f}M
Median Employee Compensation: ${random.randint(65, 120)}K
CEO Pay Ratio: {random.randint(80, 350)}:1
"""


def main() -> None:
    filings: list[dict] = []
    file_idx = 0

    for i, company in enumerate(COMPANIES):
        # Each company gets 1-3 filings of different types
        filing_types = random.sample(
            ["10-K", "8-K", "10-Q", "DEF 14A"],
            k=random.randint(1, 3),
        )

        for ftype in filing_types:
            file_idx += 1
            name_slug = company[0].lower().replace(" ", "-").replace(".", "").replace(",", "")

            if ftype == "10-K":
                content = generate_10k(company, i)
                filename = f"{file_idx:02d}-{name_slug}-10k.txt"
            elif ftype == "8-K":
                content = generate_8k(company)
                filename = f"{file_idx:02d}-{name_slug}-8k.txt"
            elif ftype == "10-Q":
                content = generate_10q(company)
                filename = f"{file_idx:02d}-{name_slug}-10q.txt"
            else:
                content = generate_def14a(company)
                filename = f"{file_idx:02d}-{name_slug}-def14a.txt"

            filepath = OUTPUT_DIR / filename
            filepath.write_text(content, encoding="utf-8")

            filings.append({
                "filename": filename,
                "company": company[0],
                "sector": company[1],
                "cik": company[2],
                "filing_type": ftype,
                "char_count": len(content),
            })

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(filings, indent=2),
        encoding="utf-8",
    )

    print(f"Generated {len(filings)} SEC filing excerpts in {OUTPUT_DIR}/")
    print(f"Manifest written to {manifest_path}")
    for f in filings:
        print(f"  {f['filename']}: {f['filing_type']} | {f['company']} | {f['char_count']:,} chars")


if __name__ == "__main__":
    random.seed(42)  # Reproducible generation
    main()
