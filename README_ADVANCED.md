# Advanced Fake News Detector

This upgraded version keeps the original project idea, but restructures the active code into smaller, human-readable modules and adds richer analysis output.

## What changed

- The core fact-checking flow is no longer trapped inside one large file.
- Claims now carry a readable profile: type, complexity, extracted entities, years, numbers, and a suggested next step.
- Reports expose evidence signals such as support score, contradiction score, dominant signal, and live source count.
- A new CLI can print either human-friendly output or full JSON.
- A new web app presents the results in a more polished, explainable interface.

## Main modules

- `app_config.py`: environment-driven runtime settings.
- `verification_models.py`: shared dataclasses for reports and evidence.
- `text_analysis.py`: claim splitting, token analysis, seed extraction, and contradiction heuristics.
- `knowledge_sources.py`: article extraction plus Google Fact Check, Wikipedia, and Wikidata lookups.
- `verification_engine.py`: orchestration layer that turns a claim or article into a structured report.
- `model_utils.py`: lightweight local classifier and cache handling.

## New entry points

Run the advanced CLI:

```bash
python cli.py
python cli.py "The government launched a new scholarship in 2024." --offline
python cli.py --url https://example.com/story --json
```

Run the advanced web app:

```bash
python web_app.py
```

Then open `http://127.0.0.1:5000`.

## Optional environment variables

- `GOOGLE_FACT_CHECK_API_KEY`: enables live Google Fact Check reviews.
- `FND_HOST`: server host for `web_app.py`.
- `FND_PORT`: server port for `web_app.py`.
- `FND_MAX_CLAIMS`: default claim cap for long inputs.
- `FND_REQUEST_TIMEOUT`: network timeout for live lookups.

## Notes about the older files

The original `front.py`, `prediction.py`, `DataPrep.py`, `FeatureSelection.py`, and `classifier.py` are still present for compatibility and historical reference.

The new architecture is built around the standard-library production flow instead of the older research-style scripts.
