from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from fact_check import analyze_document, format_verdict
from model_utils import evaluate_model, format_label, load_or_train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explainable fake-news verification with local classification, claim profiles, and optional live evidence."
    )
    parser.add_argument("text", nargs="*", help="News text to analyze. If omitted, the script prompts for it.")
    parser.add_argument(
        "--url",
        help="Optional article URL to fetch and analyze. If provided, the article text is split into individual claims.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore the cached model and train a fresh one from train.csv and valid.csv.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Show test-set accuracy before predicting.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live knowledge-source lookups and use the local model only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured report as JSON.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=None,
        help="Optional cap for how many claims to analyze from long input.",
    )
    return parser.parse_args()


def print_human_report(document) -> None:
    print(document.summary)
    print(f"Overall risk: {document.insights.overall_risk}")
    print(f"Recommended next step: {document.insights.recommended_next_step}")
    print(f"Claims analyzed: {len(document.claim_reports)}")
    print(
        "Counts:",
        f"supported={document.counts.get('supported', 0)}",
        f"contradicted={document.counts.get('contradicted', 0)}",
        f"inconclusive={document.counts.get('inconclusive', 0)}",
    )
    print(
        "Document signals:",
        f"live_claims={document.insights.live_claims}",
        f"india_related_claims={document.insights.india_related_claims}",
        f"article_mode={document.insights.article_mode}",
    )

    if document.source_url:
        print(f"Article source: {document.source_title or document.source_url}")
        print(document.source_url)

    if document.notes:
        print("Document notes:")
        for note in document.notes:
            print(f"- {note}")

    for index, report in enumerate(document.claim_reports, start=1):
        print()
        print(f"Claim {index}: {report.claim}")
        print(f"Verdict: {format_verdict(report.verdict)}")
        print(f"Confidence: {report.confidence * 100:.1f}%")
        print(f"Headline: {report.headline}")
        print(f"Summary: {report.summary}")
        print(
            "Profile:",
            f"type={report.profile.family}",
            f"complexity={report.profile.complexity}",
            f"india_related={report.profile.india_related}",
        )
        if report.profile.entities:
            print("Named entities:")
            for entity in report.profile.entities:
                print(f"- {entity}")
        if report.profile.years:
            print("Years:")
            for year in report.profile.years:
                print(f"- {year}")
        if report.profile.numbers:
            print("Numbers:")
            for number in report.profile.numbers:
                print(f"- {number}")

        print(
            "Evidence:",
            f"dominant_signal={report.evidence.dominant_signal}",
            f"live_sources={report.evidence.live_source_count}",
            f"support_score={report.evidence.support_score}",
            f"contradiction_score={report.evidence.contradiction_score}",
        )

        print("Methods:")
        for method in report.methods:
            print(f"- {method}")

        print("Reasons:")
        for reason in report.reasons:
            print(f"- {reason}")

        if report.highlights:
            print("Highlights:")
            for highlight in report.highlights:
                print(f"- {highlight}")

        if report.sources:
            print("Sources:")
            for source in report.sources:
                print(f"- [{source.source}] {source.title} :: {source.url}")
                print(f"  category: {source.category}")
                if source.detail:
                    print(f"  detail: {source.detail}")

        if report.notes:
            print("Notes:")
            for note in report.notes:
                print(f"- {note}")

        print(f"Fallback model verdict: {format_label(report.local_signal.label)}")
        print(f"Fallback model confidence: {report.local_signal.confidence * 100:.1f}%")
        print(f"Suggested next step: {report.profile.next_step}")


def main() -> None:
    args = parse_args()
    model = load_or_train_model(force_retrain=args.retrain)
    metrics = evaluate_model(model) if args.metrics else None

    text = " ".join(args.text).strip()
    if not text and not args.url:
        text = input("Please enter the news text you want to verify: ").strip()

    try:
        document = analyze_document(
            text=text,
            article_url=args.url or "",
            allow_live=not args.offline,
            model=model,
            max_claims=args.max_claims,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if args.json:
        payload = asdict(document)
        if metrics is not None:
            payload = {"metrics": metrics, "document": payload}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if metrics is not None:
        print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}% on {metrics['samples']} samples")
        print()

    print_human_report(document)


if __name__ == "__main__":
    main()
