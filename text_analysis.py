from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable
from urllib.parse import urlparse

from verification_models import AlignmentAssessment, ClaimProfile, SourceEvidence


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3}\b")
CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+|;+\s*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}
NEGATION_TERMS = {"no", "not", "never", "none", "false", "incorrect", "fake", "without", "n't"}
SUPPORTED_LABELS = ("supported", "true", "accurate", "correct", "confirmed", "mostly true")
CONTRADICTED_LABELS = (
    "false",
    "fake",
    "incorrect",
    "baseless",
    "scam",
    "misleading",
    "pants on fire",
    "not true",
    "mostly false",
)
MIXED_LABELS = ("mixture", "mixed", "half true", "partly false", "missing context", "unproven")
INDIA_TERMS = {
    "india",
    "indian",
    "bharat",
    "government",
    "govt",
    "ministry",
    "minister",
    "parliament",
    "rupee",
    "rbi",
    "pib",
    "andhra",
    "pradesh",
    "telangana",
    "delhi",
    "maharashtra",
    "karnataka",
    "kerala",
    "gujarat",
    "tamil",
    "nadu",
    "bihar",
    "odisha",
    "punjab",
    "haryana",
    "uttar",
    "madhya",
    "scheme",
    "scholarship",
    "subsidy",
    "election",
    "voter",
}
POLICY_TERMS = {
    "government",
    "policy",
    "scheme",
    "subsidy",
    "scholarship",
    "health",
    "hospital",
    "medical",
    "bank",
    "loan",
    "currency",
    "voter",
    "election",
    "poll",
}


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


@lru_cache(maxsize=512)
def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall((text or "").lower()))


@lru_cache(maxsize=512)
def meaningful_terms(text: str) -> frozenset[str]:
    return frozenset(token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS)


@lru_cache(maxsize=512)
def extract_years(text: str) -> frozenset[str]:
    return frozenset(YEAR_PATTERN.findall(text or ""))


@lru_cache(maxsize=512)
def extract_numbers(text: str) -> frozenset[str]:
    return frozenset(NUMBER_PATTERN.findall(text or ""))


@lru_cache(maxsize=512)
def extract_entities(text: str) -> tuple[str, ...]:
    matches = [
        match.group(0).strip(" .,;:!?")
        for match in ENTITY_PATTERN.finditer(text or "")
        if match.group(0).strip(" .,;:!?").lower() not in {"a", "an", "the"}
    ]
    return tuple(dedupe_strings(matches))


@lru_cache(maxsize=512)
def contains_negation(text: str) -> bool:
    return any(term in NEGATION_TERMS for term in tokenize(text))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = normalize_space(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


@lru_cache(maxsize=256)
def extract_seed_terms(text: str) -> tuple[str, ...]:
    entities = extract_entities(text)
    if entities:
        return entities

    keywords = [token for token in tokenize(text) if token not in STOPWORDS and not token.isdigit()]
    if keywords:
        return (" ".join(keywords[:4]),)

    return (normalize_space(text),)


def looks_like_url(value: str) -> bool:
    parsed = urlparse((value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_india_related_claim(claim: str) -> bool:
    return bool(set(tokenize(claim)) & INDIA_TERMS)


def map_textual_rating(rating: str) -> str:
    lowered = normalize_space(rating).lower()
    if any(label in lowered for label in CONTRADICTED_LABELS):
        return "contradicted"
    if any(label in lowered for label in MIXED_LABELS):
        return "inconclusive"
    if any(label in lowered for label in SUPPORTED_LABELS):
        return "supported"
    return "inconclusive"


def assess_reference_alignment(claim: str, evidence_text: str) -> AlignmentAssessment:
    claim_terms = meaningful_terms(claim)
    evidence_terms = meaningful_terms(evidence_text)
    overlap = jaccard(claim_terms, evidence_terms)
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence_text)
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence_text)
    support = 0.0
    contradiction = 0.0
    reasons: list[str] = []
    details: list[str] = []
    shared_terms = sorted(claim_terms & evidence_terms)[:4]

    if overlap >= 0.14:
        support += min(0.55, overlap + 0.12)
        reasons.append("Reference material discusses the same subject as the claim.")
        if shared_terms:
            details.append(f"Shared topic terms: {', '.join(shared_terms)}.")

    if claim_years and evidence_years:
        if claim_years & evidence_years:
            matching = ", ".join(sorted(claim_years & evidence_years))
            support += 0.45
            reasons.append(f"Matching year found in the reference material: {matching}.")
            details.append(f"Matching year: {matching}.")
        else:
            contradiction += 0.95
            reasons.append(
                "The claim and the reference material mention different years, which is a strong temporal contradiction."
            )
            details.append(
                f"Claim years: {', '.join(sorted(claim_years))}. Reference years: {', '.join(sorted(evidence_years))}."
            )

    non_year_claim_numbers = sorted(claim_numbers - claim_years)
    non_year_evidence_numbers = sorted(evidence_numbers - evidence_years)
    if (
        overlap >= 0.2
        and len(non_year_claim_numbers) == 1
        and len(non_year_evidence_numbers) == 1
        and non_year_claim_numbers[0] != non_year_evidence_numbers[0]
    ):
        contradiction += 0.25
        reasons.append("The claim and the reference summary use different numeric values.")
        details.append(
            f"Claim number: {non_year_claim_numbers[0]}. Reference number: {non_year_evidence_numbers[0]}."
        )

    if overlap >= 0.18 and contains_negation(claim) != contains_negation(evidence_text):
        contradiction += 0.2
        reasons.append("The claim's wording has opposite polarity from the reference summary.")

    if contradiction >= support + 0.25 and contradiction >= 0.55:
        return AlignmentAssessment(
            verdict="contradicted",
            score=min(0.96, contradiction),
            reasons=tuple(reasons),
            detail=" ".join(details),
        )

    if support > contradiction and (support >= 0.45 or (claim_years and claim_years & evidence_years)):
        return AlignmentAssessment(
            verdict="supported",
            score=min(0.92, support),
            reasons=tuple(reasons),
            detail=" ".join(details),
        )

    return AlignmentAssessment(
        verdict="inconclusive",
        score=max(support, contradiction) * 0.5,
        reasons=tuple(reasons[:1]),
        detail=" ".join(details),
    )


def split_into_claims(text: str, max_claims: int) -> list[str]:
    cleaned = normalize_space(text)
    if not cleaned:
        return []

    raw_parts = CLAIM_SPLIT_PATTERN.split(cleaned)
    claims: list[str] = []
    for part in raw_parts:
        claim = normalize_space(part.strip(" -"))
        if not claim:
            continue
        if len(claim) < 25 and len(meaningful_terms(claim)) < 4:
            continue
        claims.append(claim)

    deduped = dedupe_strings(claims)
    if not deduped:
        return [cleaned]
    return deduped[: max(1, max_claims)]


def infer_claim_family(claim: str) -> str:
    years = extract_years(claim)
    numbers = extract_numbers(claim) - years
    entities = extract_entities(claim)
    tokens = set(tokenize(claim))

    if years and numbers:
        return "Temporal / Numeric"
    if years:
        return "Temporal"
    if numbers:
        return "Numeric"
    if tokens & POLICY_TERMS or is_india_related_claim(claim):
        return "Policy / Civic"
    if entities:
        return "Entity"
    return "Narrative"


def infer_complexity(claim: str) -> str:
    signal_count = 0
    if extract_entities(claim):
        signal_count += 1
    if extract_years(claim):
        signal_count += 1
    if extract_numbers(claim):
        signal_count += 1

    term_count = len(meaningful_terms(claim))
    if term_count >= 14 or signal_count >= 3:
        return "High"
    if term_count >= 8 or signal_count >= 2:
        return "Moderate"
    return "Low"


def build_claim_profile(
    claim: str,
    verdict: str,
    allow_live: bool,
    sources: list[SourceEvidence],
) -> ClaimProfile:
    entities = list(extract_entities(claim)[:4])
    years = sorted(extract_years(claim))
    numbers = [number for number in sorted(extract_numbers(claim)) if number not in years][:4]
    india_related = is_india_related_claim(claim)
    live_source_count = sum(1 for source in sources if source.category != "official_suggestion")

    if verdict == "contradicted":
        next_step = "Open the strongest contradiction source and compare the exact wording, year, or number before sharing."
    elif verdict == "supported":
        next_step = "Confirm the publication date and original context, then share the supporting source with the claim."
    elif allow_live and live_source_count:
        next_step = "Review the listed sources and look for a primary citation that states the claim directly."
    elif allow_live:
        next_step = "Provide a more specific claim or article URL so the system can match stronger evidence."
    else:
        next_step = "Run the claim again without offline mode or provide an article URL for stronger verification."

    return ClaimProfile(
        family=infer_claim_family(claim),
        complexity=infer_complexity(claim),
        india_related=india_related,
        entities=entities,
        years=years,
        numbers=numbers,
        next_step=next_step,
    )
