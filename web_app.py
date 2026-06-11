from __future__ import annotations

import html
import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from app_config import get_runtime_config
from fact_check import DocumentReport, SourceEvidence, VerificationReport, analyze_document, format_verdict
from model_utils import format_label, load_or_train_model


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
TEMPLATE_PATH = ROOT / "templates" / "advanced_index.html"
SERVER_CONFIG = get_runtime_config()
HOST = SERVER_CONFIG.host
PORT = SERVER_CONFIG.port


@dataclass(frozen=True)
class HttpResponse:
    body: bytes
    status: HTTPStatus
    content_type: str = "text/html; charset=utf-8"

    @property
    def headers(self) -> list[tuple[str, str]]:
        return [
            ("Content-Type", self.content_type),
            ("Content-Length", str(len(self.body))),
        ]


@dataclass(frozen=True)
class RequestPayload:
    news_text: str
    article_url: str


@lru_cache(maxsize=1)
def load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def render_page(
    news_text: str = "",
    article_url: str = "",
    result_html: str = "",
    error_text: str = "",
) -> bytes:
    template = load_template()
    page = (
        template.replace("__NEWS_TEXT__", html.escape(news_text))
        .replace("__ARTICLE_URL__", html.escape(article_url))
        .replace("__RESULT_BLOCK__", result_html)
        .replace("__ERROR_TEXT__", html.escape(error_text))
    )
    return page.encode("utf-8")


def render_list_items(items: list[str]) -> str:
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def render_tags(items: list[str], tone: str = "") -> str:
    if not items:
        return ""
    tone_class = f" tag--{tone}" if tone else ""
    return "".join(f"<span class='tag{tone_class}'>{html.escape(item)}</span>" for item in items)


def render_metadata_row(title: str, items: list[str], tone: str = "") -> str:
    if not items:
        return ""
    return f"""
    <div class="metadata-row">
      <span class="metadata-row__label">{html.escape(title)}</span>
      <div class="tag-row">{render_tags(items, tone)}</div>
    </div>
    """


def format_source_category(category: str) -> str:
    labels = {
        "fact_check_review": "Fact-check review",
        "knowledge_reference": "Reference summary",
        "knowledge_graph": "Knowledge graph",
        "official_suggestion": "Official source idea",
    }
    return labels.get(category, category.replace("_", " ").title())


def render_source_card(source: SourceEvidence) -> str:
    detail_html = f"<p class='source-card__detail'>{html.escape(source.detail)}</p>" if source.detail else ""
    verdict = format_verdict(source.verdict)
    return f"""
    <article class="source-card">
      <div class="source-card__header">
        <p class="source-card__meta">{html.escape(source.source)}</p>
        <span class="tag tag--neutral">{html.escape(format_source_category(source.category))}</span>
      </div>
      <h4>{html.escape(source.title)}</h4>
      <p>{html.escape(source.snippet)}</p>
      {detail_html}
      <div class="source-card__footer">
        <span>{html.escape(verdict)}</span>
        <a href="{html.escape(source.url)}" target="_blank" rel="noopener noreferrer">Open source</a>
      </div>
    </article>
    """


def render_claim_card(report: VerificationReport, index: int) -> str:
    verdict = format_verdict(report.verdict)
    confidence = f"{report.confidence * 100:.1f}%"
    local_label = format_label(report.local_signal.label)
    local_confidence = f"{report.local_signal.confidence * 100:.1f}%"
    tone_map = {
        "supported": "claim-card--true",
        "contradicted": "claim-card--false",
        "inconclusive": "claim-card--neutral",
    }
    tone = tone_map.get(report.verdict, "claim-card--neutral")
    methods_html = render_list_items(report.methods)
    reasons_html = render_list_items(report.reasons)
    highlights_html = render_list_items(report.highlights) if report.highlights else ""
    notes_html = render_list_items(report.notes) if report.notes else ""
    sources_html = "".join(render_source_card(source) for source in report.sources if source.url)

    profile_tags = [
        f"Type: {report.profile.family}",
        f"Complexity: {report.profile.complexity}",
    ]
    if report.profile.india_related:
        profile_tags.append("India-related")

    metadata_html = "".join(
        [
            render_metadata_row("Claim profile", profile_tags, "accent"),
            render_metadata_row("Named entities", report.profile.entities),
            render_metadata_row("Years", report.profile.years),
            render_metadata_row("Numbers", report.profile.numbers),
        ]
    )

    return f"""
    <section class="claim-card {tone}">
      <div class="claim-card__header">
        <p class="claim-card__eyebrow">Claim {index}</p>
        <h3>{html.escape(report.claim)}</h3>
      </div>
      <p class="claim-card__headline">{html.escape(report.headline)}</p>
      <p class="claim-card__summary">{html.escape(report.summary)}</p>
      <div class="stat-grid">
        <div>
          <span>Verdict</span>
          <strong>{html.escape(verdict)}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>{confidence}</strong>
        </div>
        <div>
          <span>Dominant signal</span>
          <strong>{html.escape(report.evidence.dominant_signal)}</strong>
        </div>
        <div>
          <span>Live sources</span>
          <strong>{report.evidence.live_source_count}</strong>
        </div>
        <div>
          <span>Local model</span>
          <strong>{html.escape(local_label)}</strong>
        </div>
        <div>
          <span>Local confidence</span>
          <strong>{local_confidence}</strong>
        </div>
      </div>
      <div class="claim-card__section">
        <h4>Profile</h4>
        {metadata_html or "<p class='empty-state'>No extra claim markers were extracted.</p>"}
      </div>
      <div class="claim-card__section">
        <h4>How it checked the claim</h4>
        <ul class="detail-list">{methods_html}</ul>
      </div>
      <div class="claim-card__section">
        <h4>Why this verdict</h4>
        <ul class="detail-list">{reasons_html}</ul>
      </div>
      {f"<div class='claim-card__section'><h4>Key highlights</h4><ul class='detail-list'>{highlights_html}</ul></div>" if highlights_html else ""}
      <div class="claim-card__section">
        <h4>Suggested next step</h4>
        <p class="next-step">{html.escape(report.profile.next_step)}</p>
      </div>
      <div class="claim-card__section">
        <h4>Reference sources</h4>
        <div class="source-grid">
          {sources_html or "<p class='empty-state'>No live source card was available for this claim.</p>"}
        </div>
      </div>
      {f"<div class='claim-card__section'><h4>Notes</h4><ul class='detail-list'>{notes_html}</ul></div>" if notes_html else ""}
    </section>
    """


def build_result_block(document: DocumentReport) -> str:
    counts = document.counts
    claim_cards = "".join(
        render_claim_card(report, index)
        for index, report in enumerate(document.claim_reports, start=1)
    )
    source_html = ""
    if document.source_url:
        label = html.escape(document.source_title or document.source_url)
        source_html = (
            f"<p class='document-result__source'>Article source: "
            f"<a href='{html.escape(document.source_url)}' target='_blank' rel='noopener noreferrer'>{label}</a></p>"
        )

    notes_html = render_list_items(document.notes) if document.notes else ""
    summary_tags = [
        f"Overall risk: {document.insights.overall_risk}",
        f"Live-backed claims: {document.insights.live_claims}",
        f"India-related claims: {document.insights.india_related_claims}",
    ]
    if document.insights.article_mode:
        summary_tags.append("Article mode")

    return f"""
    <section class="document-result">
      <div class="document-result__intro">
        <p class="result__eyebrow">Document summary</p>
        <h2>{html.escape(document.summary)}</h2>
        <div class="tag-row">{render_tags(summary_tags, 'neutral')}</div>
        {source_html}
      </div>
      <div class="stat-grid stat-grid--document">
        <div>
          <span>Claims analyzed</span>
          <strong>{len(document.claim_reports)}</strong>
        </div>
        <div>
          <span>Supported</span>
          <strong>{counts.get('supported', 0)}</strong>
        </div>
        <div>
          <span>Contradicted</span>
          <strong>{counts.get('contradicted', 0)}</strong>
        </div>
        <div>
          <span>Need evidence</span>
          <strong>{counts.get('inconclusive', 0)}</strong>
        </div>
      </div>
      <div class="document-result__section">
        <h3>Recommended next step</h3>
        <p class="next-step next-step--document">{html.escape(document.insights.recommended_next_step)}</p>
      </div>
      {f"<div class='document-result__section'><h3>Document notes</h3><ul class='detail-list'>{notes_html}</ul></div>" if notes_html else ""}
      <div class="claim-stack">
        {claim_cards}
      </div>
    </section>
    """


def html_response(
    news_text: str = "",
    article_url: str = "",
    result_html: str = "",
    error_text: str = "",
) -> HttpResponse:
    return HttpResponse(
        body=render_page(news_text=news_text, article_url=article_url, result_html=result_html, error_text=error_text),
        status=HTTPStatus.OK,
    )


def plain_text_response(text: str, status: HTTPStatus) -> HttpResponse:
    return HttpResponse(body=text.encode("utf-8"), status=status, content_type="text/plain; charset=utf-8")


def serve_static(path: str) -> HttpResponse:
    requested = (STATIC_DIR / path.removeprefix("/static/")).resolve()

    try:
        requested.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return plain_text_response("Not Found", HTTPStatus.NOT_FOUND)

    if not requested.is_file():
        return plain_text_response("Not Found", HTTPStatus.NOT_FOUND)

    content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
    return HttpResponse(body=requested.read_bytes(), status=HTTPStatus.OK, content_type=content_type)


def read_request_body(environ) -> bytes:
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        content_length = 0
    return environ["wsgi.input"].read(content_length)


def parse_request_payload(environ) -> RequestPayload:
    body = read_request_body(environ)
    form_data = parse_qs(body.decode("utf-8", errors="replace"))
    return RequestPayload(
        news_text=form_data.get("news", [""])[0],
        article_url=form_data.get("article_url", [""])[0],
    )


def handle_prediction(environ) -> HttpResponse:
    payload = parse_request_payload(environ)

    try:
        document = analyze_document(
            text=payload.news_text,
            article_url=payload.article_url,
        )
        result_html = build_result_block(document)
        return html_response(news_text=payload.news_text, article_url=payload.article_url, result_html=result_html)
    except ValueError as exc:
        return html_response(news_text=payload.news_text, article_url=payload.article_url, error_text=str(exc))
    except Exception as exc:  # pragma: no cover - defensive path for manual runs
        return html_response(
            news_text=payload.news_text,
            article_url=payload.article_url,
            error_text=f"Unable to analyze the text: {exc}",
        )


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if path == "/static/advanced.css":
        response = serve_static(path)
        start_response(f"{response.status.value} {response.status.phrase}", response.headers)
        return [response.body]

    if path == "/" and method == "GET":
        response = html_response()
        start_response(f"{response.status.value} {response.status.phrase}", response.headers)
        return [response.body]

    if path == "/predict" and method == "POST":
        response = handle_prediction(environ)
        start_response(f"{response.status.value} {response.status.phrase}", response.headers)
        return [response.body]

    response = plain_text_response("Not Found", HTTPStatus.NOT_FOUND)
    start_response(f"{response.status.value} {response.status.phrase}", response.headers)
    return [response.body]


def main() -> None:
    load_or_train_model()
    print(f"Advanced Fake News Detector running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        with make_server(HOST, PORT, app) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
