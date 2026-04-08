class FilingClassification(BaseModel):
    filing_type: str      # 10-K, 10-Q, 8-K, DEF 14A, S-1, UNKNOWN
    confidence: float     # 0.0 to 1.0
    evidence: list[str]   # Phrases from the text supporting the classification
    ambiguity_notes: str | None  # Explain uncertainty, or null