from __future__ import annotations

from dataclasses import dataclass

from model_utils import PredictionResult


@dataclass(frozen=True)
class SourceEvidence:
    source: str
    title: str
    url: str
    snippet: str
    verdict: str = "inconclusive"
    score: float = 0.0
    rating: str = ""
    published_on: str = ""
    detail: str = ""
    category: str = "reference"


@dataclass(frozen=True)
class AlignmentAssessment:
    verdict: str
    score: float
    reasons: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class ArticleExtraction:
    url: str
    title: str
    text: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ClaimProfile:
    family: str
    complexity: str
    india_related: bool
    entities: list[str]
    years: list[str]
    numbers: list[str]
    next_step: str


@dataclass(frozen=True)
class EvidenceSignals:
    support_score: float
    contradiction_score: float
    source_count: int
    live_source_count: int
    dominant_signal: str


@dataclass(frozen=True)
class VerificationReport:
    claim: str
    verdict: str
    confidence: float
    headline: str
    summary: str
    reasons: list[str]
    methods: list[str]
    sources: list[SourceEvidence]
    highlights: list[str]
    notes: list[str]
    local_signal: PredictionResult
    profile: ClaimProfile
    evidence: EvidenceSignals


@dataclass(frozen=True)
class DocumentInsights:
    overall_risk: str
    article_mode: bool
    live_claims: int
    india_related_claims: int
    recommended_next_step: str


@dataclass(frozen=True)
class DocumentReport:
    original_input: str
    source_url: str
    source_title: str
    prepared_text: str
    claim_reports: list[VerificationReport]
    counts: dict[str, int]
    summary: str
    notes: list[str]
    insights: DocumentInsights
