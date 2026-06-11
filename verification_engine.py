from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app_config import RuntimeConfig, get_runtime_config
from knowledge_sources import (
    build_india_source_suggestions,
    fetch_article,
    fetch_google_fact_checks,
    fetch_wikidata_entities,
    fetch_wikipedia_references,
)
from model_utils import PredictionResult, TrainedModel, load_or_train_model, predict_text
from text_analysis import (
    assess_reference_alignment,
    build_claim_profile,
    dedupe_strings,
    extract_seed_terms,
    looks_like_url,
    normalize_space,
    split_into_claims,
)
from verification_models import (
    DocumentInsights,
    DocumentReport,
    EvidenceSignals,
    SourceEvidence,
    VerificationReport,
)


def _filter_display_notes(notes: list[str]) -> list[str]:
    hidden_prefixes = (
        "Set GOOGLE_FACT_CHECK_API_KEY",
        "The local model below is a secondary signal",
        "Live knowledge-source lookups were disabled",
        "Live knowledge sources were unavailable",
    )
    return [note for note in notes if not note.startswith(hidden_prefixes)]


def _extract_highlights(sources: list[SourceEvidence]) -> list[str]:
    highlights: list[str] = []
    for source in sources:
        if source.detail and source.verdict in {"contradicted", "supported"}:
            prefix = "Conflict" if source.verdict == "contradicted" else "Match"
            highlights.append(f"{prefix} from {source.source}: {source.detail}")
    return dedupe_strings(highlights)[:4]


def _build_evidence_signals(
    support_score: float,
    contradiction_score: float,
    sources: list[SourceEvidence],
    local_signal: PredictionResult,
) -> EvidenceSignals:
    live_source_count = sum(1 for source in sources if source.category != "official_suggestion")

    if contradiction_score > support_score and contradiction_score >= 0.55:
        dominant_signal = "Contradiction evidence"
    elif support_score > contradiction_score and support_score >= 0.45:
        dominant_signal = "Support evidence"
    elif live_source_count:
        dominant_signal = "Weak or mixed evidence"
    else:
        dominant_signal = f"Local model leaning {local_signal.label.title()}"

    return EvidenceSignals(
        support_score=round(support_score, 3),
        contradiction_score=round(contradiction_score, 3),
        source_count=len(sources),
        live_source_count=live_source_count,
        dominant_signal=dominant_signal,
    )


def _build_report(
    claim: str,
    local_signal: PredictionResult,
    google_evidence: list[SourceEvidence],
    wikipedia_evidence: list[SourceEvidence],
    wikidata_evidence: list[SourceEvidence],
    notes: list[str],
    allow_live: bool,
) -> VerificationReport:
    methods: list[str] = []
    reasons: list[str] = []
    sources: list[SourceEvidence] = []
    support_score = 0.0
    contradiction_score = 0.0

    if google_evidence:
        methods.append("Google Fact Check Tools API")
        sources.extend(google_evidence)
        contradicted_reviews = [item for item in google_evidence if item.verdict == "contradicted"]
        supported_reviews = [item for item in google_evidence if item.verdict == "supported"]

        support_score += sum(item.score for item in supported_reviews)
        contradiction_score += sum(item.score for item in contradicted_reviews)

        if contradicted_reviews:
            reasons.append("Live fact-check reviews rated similar claims as false, misleading, or incorrect.")
        elif supported_reviews:
            reasons.append("Live fact-check reviews rated similar claims as true or supported.")

    assessed_wikipedia: list[SourceEvidence] = []
    if wikipedia_evidence:
        methods.append("Wikipedia reference summaries")
        for item in wikipedia_evidence:
            assessment = assess_reference_alignment(claim, item.snippet)
            assessed_wikipedia.append(
                SourceEvidence(
                    source=item.source,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    verdict=assessment.verdict,
                    score=assessment.score,
                    rating=item.rating,
                    published_on=item.published_on,
                    detail=assessment.detail,
                    category=item.category,
                )
            )

            if assessment.verdict == "supported":
                support_score += assessment.score
            elif assessment.verdict == "contradicted":
                contradiction_score += assessment.score

            reasons.extend(assessment.reasons[:1])

        sources.extend(assessed_wikipedia)

    if wikidata_evidence:
        methods.append("Wikidata entity graph")
        sources.extend(wikidata_evidence)
        reasons.append("Wikidata entity matches were used to anchor the claim to known people, places, or events.")

    india_sources = build_india_source_suggestions(claim)
    if india_sources:
        methods.append("India-specific official source suggestions")
        sources.extend(india_sources)

    methods = dedupe_strings(methods)
    reasons = dedupe_strings(reasons)[:4]
    notes = _filter_display_notes(dedupe_strings(notes))
    sources = sources[:8]
    highlights = _extract_highlights(sources)
    evidence = _build_evidence_signals(support_score, contradiction_score, sources, local_signal)

    knowledge_strength = max(support_score, contradiction_score)
    knowledge_total = support_score + contradiction_score

    if knowledge_strength >= 0.75 and abs(support_score - contradiction_score) >= 0.25:
        if contradiction_score > support_score:
            verdict = "contradicted"
            headline = "Likely contradicted by reference sources"
            winning_score = contradiction_score
        else:
            verdict = "supported"
            headline = "Supported by reference sources"
            winning_score = support_score

        confidence = max(0.55, min(0.97, winning_score / max(knowledge_total, 1.0)))
        summary = "This verdict is driven mainly by live reference lookups, entity grounding, and contradiction checks."
    else:
        verdict = "inconclusive"
        headline = "No strong factual match found"
        confidence = max(0.35, min(0.7, 0.4 + (knowledge_strength * 0.15)))
        if allow_live:
            summary = (
                "The system did not find enough direct evidence to make a strong fact verdict, so the local model is shown only as a fallback signal."
            )
        else:
            summary = "Live lookups were disabled, so this run relies on local language patterns and claim structure hints only."

    profile = build_claim_profile(claim, verdict, allow_live, sources)

    return VerificationReport(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        headline=headline,
        summary=summary,
        reasons=reasons or ["No decisive reference match was found for this claim."],
        methods=methods or ["Local language-pattern model"],
        sources=sources,
        highlights=highlights,
        notes=notes,
        local_signal=local_signal,
        profile=profile,
        evidence=evidence,
    )


def analyze_claim(
    text: str,
    allow_live: bool = True,
    model: TrainedModel | None = None,
    config: RuntimeConfig | None = None,
) -> VerificationReport:
    cleaned = normalize_space(text)
    if not cleaned:
        raise ValueError("Please enter some news text first.")

    active_config = config or get_runtime_config()
    local_signal = predict_text(cleaned, model or load_or_train_model())
    notes: list[str] = []
    google_evidence: list[SourceEvidence] = []
    wikipedia_evidence: list[SourceEvidence] = []
    wikidata_evidence: list[SourceEvidence] = []

    if allow_live:
        with ThreadPoolExecutor(max_workers=2) as executor:
            google_future = executor.submit(fetch_google_fact_checks, cleaned, active_config)
            wikipedia_future = executor.submit(fetch_wikipedia_references, cleaned, active_config)

            google_evidence, google_notes = google_future.result()
            wikipedia_evidence, wikipedia_notes = wikipedia_future.result()

        notes.extend(google_notes)
        notes.extend(wikipedia_notes)

        seed_terms = [item.title for item in wikipedia_evidence]
        if not seed_terms:
            seed_terms = list(extract_seed_terms(cleaned))
        wikidata_evidence, wikidata_notes = fetch_wikidata_entities(seed_terms, active_config)
        notes.extend(wikidata_notes)
    else:
        notes.append("Live knowledge-source lookups were disabled for this run.")

    return _build_report(
        claim=cleaned,
        local_signal=local_signal,
        google_evidence=google_evidence,
        wikipedia_evidence=wikipedia_evidence,
        wikidata_evidence=wikidata_evidence,
        notes=notes,
        allow_live=allow_live,
    )


def summarize_document_reports(reports: list[VerificationReport]) -> tuple[dict[str, int], str]:
    counts = {"supported": 0, "contradicted": 0, "inconclusive": 0}
    for report in reports:
        counts[report.verdict] = counts.get(report.verdict, 0) + 1

    if counts["contradicted"]:
        summary = f"Potential factual issues found in {counts['contradicted']} of {len(reports)} analyzed claim(s)."
    elif counts["supported"] == len(reports):
        summary = "All analyzed claims were supported by the available evidence."
    else:
        summary = "Some claims could not be verified strongly and need more evidence."

    return counts, summary


def _build_document_insights(
    reports: list[VerificationReport],
    source_url: str,
    allow_live: bool,
) -> DocumentInsights:
    live_claims = sum(1 for report in reports if report.evidence.live_source_count)
    india_related_claims = sum(1 for report in reports if report.profile.india_related)
    high_risk_claims = sum(1 for report in reports if report.verdict == "contradicted" and report.confidence >= 0.7)
    contradicted_claims = sum(1 for report in reports if report.verdict == "contradicted")

    if high_risk_claims:
        overall_risk = "High"
        next_step = "Review the contradicted claims first and compare them against the strongest linked source before sharing."
    elif contradicted_claims:
        overall_risk = "Guarded"
        next_step = "Check the claims marked as contradicted, especially where the system found conflicting dates or numbers."
    elif all(report.verdict == "supported" for report in reports):
        overall_risk = "Low"
        next_step = "The claims look healthy, but it is still worth checking date context and the original article framing."
    elif allow_live:
        overall_risk = "Needs review"
        next_step = "Provide a more specific article or claim wording to help the verifier find stronger evidence."
    else:
        overall_risk = "Offline review"
        next_step = "Enable live verification for stronger evidence, or provide an article URL with more context."

    return DocumentInsights(
        overall_risk=overall_risk,
        article_mode=bool(source_url),
        live_claims=live_claims,
        india_related_claims=india_related_claims,
        recommended_next_step=next_step,
    )


def analyze_document(
    text: str = "",
    article_url: str = "",
    allow_live: bool = True,
    model: TrainedModel | None = None,
    max_claims: int | None = None,
    config: RuntimeConfig | None = None,
) -> DocumentReport:
    active_config = config or get_runtime_config()
    base_text = normalize_space(text)
    url = article_url.strip() if article_url else ""
    notes: list[str] = []
    source_title = ""
    source_url = ""

    if not url and looks_like_url(base_text):
        url = base_text
        base_text = ""

    if url:
        article = fetch_article(url, active_config)
        source_url = article.url
        source_title = article.title
        if not base_text:
            base_text = article.text
        else:
            base_text = f"{base_text} {article.text}"
        notes.extend(article.notes)

    if not base_text:
        raise ValueError("Please enter some news text or provide an article URL.")

    shared_model = model or load_or_train_model()
    claim_limit = max(1, max_claims or active_config.max_claims)
    claims = split_into_claims(base_text, max_claims=claim_limit)
    claim_reports = [
        analyze_claim(claim, allow_live=allow_live, model=shared_model, config=active_config)
        for claim in claims
    ]

    total_candidates = len(split_into_claims(base_text, max_claims=claim_limit + 20))
    if total_candidates > len(claim_reports):
        notes.append(f"Only the first {claim_limit} claims were analyzed to keep the response focused.")

    counts, summary = summarize_document_reports(claim_reports)
    insights = _build_document_insights(claim_reports, source_url, allow_live)
    return DocumentReport(
        original_input=text or article_url,
        source_url=source_url,
        source_title=source_title,
        prepared_text=base_text,
        claim_reports=claim_reports,
        counts=counts,
        summary=summary,
        notes=dedupe_strings(notes),
        insights=insights,
    )


def format_verdict(verdict: str) -> str:
    mapping = {
        "supported": "Supported",
        "contradicted": "Contradicted",
        "inconclusive": "Needs more evidence",
    }
    return mapping.get(verdict, verdict.replace("_", " ").title())
