from __future__ import annotations

import json
from functools import lru_cache
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from app_config import RuntimeConfig, get_runtime_config
from text_analysis import (
    dedupe_strings,
    extract_seed_terms,
    is_india_related_claim,
    map_textual_rating,
    normalize_space,
    tokenize,
)
from verification_models import ArticleExtraction, SourceEvidence


INDIA_HEALTH_TERMS = {"health", "hospital", "doctor", "vaccine", "medical", "disease", "covid", "clinic"}
INDIA_FINANCE_TERMS = {"bank", "loan", "rupee", "rbi", "inflation", "interest", "currency", "upi", "finance"}
INDIA_ELECTION_TERMS = {"election", "vote", "voter", "poll", "evm", "ballot", "booth", "constituency"}
INDIA_SCHEME_TERMS = {"scheme", "subsidy", "scholarship", "benefit", "education", "farmer", "ration", "pension"}


def _fetch_json(url: str, config: RuntimeConfig) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": config.user_agent,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=config.request_timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class ArticleHTMLParser(HTMLParser):
    BLOCK_TAGS = {"p", "li", "blockquote", "h1", "h2", "h3"}
    SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._current_block: list[str] | None = None
        self._title_parts: list[str] = []
        self._in_title = False
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self._current_block = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS and self._current_block is not None:
            text = normalize_space(" ".join(self._current_block))
            if len(text) >= 40:
                self.blocks.append(text)
            self._current_block = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        if self._current_block is not None:
            self._current_block.append(data)

    @property
    def title(self) -> str:
        return normalize_space(" ".join(self._title_parts))


@lru_cache(maxsize=32)
def _fetch_article_cached(url: str, config: RuntimeConfig) -> ArticleExtraction:
    request = Request(
        url,
        headers={
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    notes: list[str] = []

    try:
        with urlopen(request, timeout=config.request_timeout + 1.5) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html_text = response.read().decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise ValueError(f"Unable to fetch the article URL: {exc}") from exc

    parser = ArticleHTMLParser()
    parser.feed(html_text)
    parser.close()

    blocks = dedupe_strings(parser.blocks)
    article_text = " ".join(blocks[:12])
    if not article_text:
        raise ValueError("The article URL was fetched, but no readable article text could be extracted.")

    if len(blocks) > 12:
        notes.append("Only the first part of the article was analyzed to keep the fact-check focused.")

    return ArticleExtraction(
        url=url,
        title=parser.title or url,
        text=article_text,
        notes=tuple(notes),
    )


def fetch_article(url: str, config: RuntimeConfig | None = None) -> ArticleExtraction:
    return _fetch_article_cached(url, config or get_runtime_config())


@lru_cache(maxsize=64)
def _fetch_google_fact_checks_cached(claim: str, config: RuntimeConfig) -> tuple[tuple[SourceEvidence, ...], tuple[str, ...]]:
    if not config.google_fact_check_api_key:
        return (), ("Set GOOGLE_FACT_CHECK_API_KEY to enable live Google Fact Check reviews.",)

    params = urlencode(
        {
            "query": claim,
            "languageCode": "en-US",
            "pageSize": config.fact_check_page_size,
            "key": config.google_fact_check_api_key,
        }
    )
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{params}"

    try:
        payload = _fetch_json(url, config)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return (), (f"Google Fact Check lookup failed: {exc}",)

    evidence: list[SourceEvidence] = []
    for claim_item in payload.get("claims", []):
        reviewed_claim = normalize_space(claim_item.get("text", ""))
        for review in claim_item.get("claimReview", []):
            publisher = (review.get("publisher") or {}).get("name") or "Fact check review"
            rating = normalize_space(review.get("textualRating", ""))
            verdict = map_textual_rating(rating)
            review_title = normalize_space(review.get("title", "")) or reviewed_claim or "Claim review"
            snippet = reviewed_claim or review_title
            detail = normalize_space(
                " ".join(
                    part
                    for part in [
                        f"Rating: {rating}." if rating else "",
                        f"Published: {normalize_space(review.get('reviewDate', ''))}."
                        if review.get("reviewDate")
                        else "",
                    ]
                    if part
                )
            )
            evidence.append(
                SourceEvidence(
                    source=publisher,
                    title=review_title,
                    url=review.get("url", ""),
                    snippet=snippet,
                    verdict=verdict,
                    score=1.8 if verdict in {"supported", "contradicted"} else 0.8,
                    rating=rating,
                    published_on=normalize_space(review.get("reviewDate", "")),
                    detail=detail,
                    category="fact_check_review",
                )
            )

    return tuple(evidence[: config.fact_check_page_size]), ()


def fetch_google_fact_checks(claim: str, config: RuntimeConfig | None = None) -> tuple[list[SourceEvidence], list[str]]:
    evidence, notes = _fetch_google_fact_checks_cached(claim, config or get_runtime_config())
    return list(evidence), list(notes)


@lru_cache(maxsize=64)
def _fetch_wikipedia_references_cached(
    claim: str, config: RuntimeConfig
) -> tuple[tuple[SourceEvidence, ...], tuple[str, ...]]:
    queries = (claim,) + extract_seed_terms(claim)
    search_errors: list[str] = []
    titles: list[str] = []

    for query in dedupe_strings(queries):
        search_params = urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": config.wikipedia_page_size,
                "format": "json",
                "utf8": 1,
            }
        )
        search_url = f"https://en.wikipedia.org/w/api.php?{search_params}"

        try:
            search_payload = _fetch_json(search_url, config)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            search_errors.append(f"Wikipedia lookup failed: {exc}")
            continue

        titles = [
            item.get("title", "")
            for item in search_payload.get("query", {}).get("search", [])
            if item.get("title")
        ]
        if titles:
            break

    if not titles:
        notes = search_errors or ["Wikipedia did not find a close reference page for this claim."]
        return (), tuple(notes)

    extract_params = urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "titles": "|".join(titles),
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "utf8": 1,
        }
    )
    extract_url = f"https://en.wikipedia.org/w/api.php?{extract_params}"

    try:
        extract_payload = _fetch_json(extract_url, config)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return (), (f"Wikipedia extract lookup failed: {exc}",)

    pages = extract_payload.get("query", {}).get("pages", {})
    title_order = {title: index for index, title in enumerate(titles)}
    ordered_pages = sorted(
        pages.values(),
        key=lambda page: title_order.get(page.get("title", ""), len(title_order)),
    )

    evidence: list[SourceEvidence] = []
    for page in ordered_pages:
        title = normalize_space(page.get("title", ""))
        extract = normalize_space(page.get("extract", ""))
        if not title or not extract:
            continue
        evidence.append(
            SourceEvidence(
                source="Wikipedia",
                title=title,
                url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                snippet=extract[:420],
                category="knowledge_reference",
            )
        )

    return tuple(evidence[: config.wikipedia_page_size]), ()


def fetch_wikipedia_references(claim: str, config: RuntimeConfig | None = None) -> tuple[list[SourceEvidence], list[str]]:
    evidence, notes = _fetch_wikipedia_references_cached(claim, config or get_runtime_config())
    return list(evidence), list(notes)


@lru_cache(maxsize=64)
def _fetch_wikidata_entities_cached(
    seed_terms: tuple[str, ...],
    config: RuntimeConfig,
) -> tuple[tuple[SourceEvidence, ...], tuple[str, ...]]:
    evidence: list[SourceEvidence] = []
    notes: list[str] = []
    seen_ids: set[str] = set()

    for term in seed_terms[: config.wikidata_page_size]:
        if not term:
            continue

        params = urlencode(
            {
                "action": "wbsearchentities",
                "search": term,
                "language": "en",
                "format": "json",
                "limit": 1,
            }
        )
        url = f"https://www.wikidata.org/w/api.php?{params}"

        try:
            payload = _fetch_json(url, config)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            notes.append(f"Wikidata lookup failed for '{term}': {exc}")
            continue

        for item in payload.get("search", []):
            entity_id = item.get("id", "")
            if not entity_id or entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            label = normalize_space(item.get("label", "")) or entity_id
            description = normalize_space(item.get("description", "")) or "Knowledge graph entity match."
            evidence.append(
                SourceEvidence(
                    source="Wikidata",
                    title=f"{label} ({entity_id})",
                    url=item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}",
                    snippet=description,
                    score=0.2,
                    detail=description,
                    category="knowledge_graph",
                )
            )

    return tuple(evidence[: config.wikidata_page_size]), tuple(notes)


def fetch_wikidata_entities(
    seed_terms: list[str],
    config: RuntimeConfig | None = None,
) -> tuple[list[SourceEvidence], list[str]]:
    evidence, notes = _fetch_wikidata_entities_cached(tuple(seed_terms), config or get_runtime_config())
    return list(evidence), list(notes)


def build_india_source_suggestions(claim: str) -> list[SourceEvidence]:
    if not is_india_related_claim(claim):
        return []

    query = quote(claim)
    tokens = set(tokenize(claim))
    suggestions = [
        SourceEvidence(
            source="PIB Fact Check",
            title="Open PIB Fact Check portal",
            url="https://factcheck.pib.gov.in/",
            snippet="Use the Government of India's fact-check portal for claims about government announcements or viral posts.",
            detail="Useful for India-specific government misinformation and public policy rumors.",
            category="official_suggestion",
        )
    ]

    if tokens & INDIA_HEALTH_TERMS:
        suggestions.append(
            SourceEvidence(
                source="MoHFW",
                title="Search Ministry of Health and Family Welfare",
                url=f"https://www.mohfw.gov.in/search/node/{query}",
                snippet="Check official ministry releases and health advisories.",
                detail="Recommended for health, vaccine, disease, and hospital-related claims.",
                category="official_suggestion",
            )
        )

    if tokens & INDIA_FINANCE_TERMS:
        suggestions.append(
            SourceEvidence(
                source="Reserve Bank of India",
                title="Search RBI official site",
                url=f"https://www.rbi.org.in/scripts/SearchResults.aspx?search={query}",
                snippet="Check RBI notifications, circulars, and FAQs for finance-related claims.",
                detail="Recommended for banking, loan, currency, and UPI claims.",
                category="official_suggestion",
            )
        )

    if tokens & INDIA_ELECTION_TERMS:
        suggestions.append(
            SourceEvidence(
                source="Election Commission of India",
                title="Open Election Commission of India",
                url="https://www.eci.gov.in/",
                snippet="Check the ECI portal for election schedules, rules, and official announcements.",
                detail="Recommended for voter, poll, and EVM-related claims.",
                category="official_suggestion",
            )
        )

    if tokens & INDIA_SCHEME_TERMS:
        suggestions.append(
            SourceEvidence(
                source="myScheme",
                title="Search Indian government schemes",
                url="https://www.myscheme.gov.in/find-scheme",
                snippet="Use the national scheme discovery platform for subsidy, scholarship, and welfare claims.",
                detail="Recommended for benefits, education, subsidy, pension, and welfare claims.",
                category="official_suggestion",
            )
        )

    return suggestions[:4]
