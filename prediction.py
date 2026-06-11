from __future__ import annotations

import argparse

from fact_check import analyze_document, format_verdict
from model_utils import evaluate_model, format_label, load_or_train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify whether a news statement looks real or fake.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_or_train_model(force_retrain=args.retrain)

    if args.metrics:
        metrics = evaluate_model(model)
        print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}% on {metrics['samples']} samples")

    text = " ".join(args.text).strip()
    if not text and not args.url:
        text = input("Please enter the news text you want to verify: ").strip()

    try:
        document = analyze_document(
            text=text,
            article_url=args.url or "",
            allow_live=not args.offline,
            model=model,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print(document.summary)
    print(f"Claims analyzed: {len(document.claim_reports)}")
    print(
        "Counts:",
        f"supported={document.counts.get('supported', 0)}",
        f"contradicted={document.counts.get('contradicted', 0)}",
        f"inconclusive={document.counts.get('inconclusive', 0)}",
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
        print(f"Summary: {report.summary}")
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
                if source.detail:
                    print(f"  detail: {source.detail}")

        if report.notes:
            print("Notes:")
            for note in report.notes:
                print(f"- {note}")

        print(f"Fallback model verdict: {format_label(report.local_signal.label)}")
        print(f"Fallback model confidence: {report.local_signal.confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
