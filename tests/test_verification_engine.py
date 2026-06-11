from __future__ import annotations

import unittest

from text_analysis import assess_reference_alignment, split_into_claims
from verification_engine import analyze_document, format_verdict


class VerificationEngineTests(unittest.TestCase):
    def test_split_into_claims_respects_limit(self) -> None:
        text = (
            "India launched a scholarship for students in 2024. "
            "The central bank changed a currency rule in the same year. "
            "A third long-form claim appears here for trimming."
        )
        claims = split_into_claims(text, max_claims=2)
        self.assertEqual(len(claims), 2)

    def test_alignment_detects_temporal_conflict(self) -> None:
        assessment = assess_reference_alignment(
            "The event happened in 2024 and involved 50 participants.",
            "The event happened in 2022 and involved 50 participants.",
        )
        self.assertEqual(assessment.verdict, "contradicted")

    def test_offline_document_analysis_returns_structured_profiles(self) -> None:
        document = analyze_document(
            text=(
                "India launched a scholarship for 5000 students in 2024. "
                "Another detailed claim says a new banking rule changed payment limits this year."
            ),
            allow_live=False,
            max_claims=2,
        )
        self.assertEqual(len(document.claim_reports), 2)
        self.assertTrue(document.insights.overall_risk)
        self.assertIn(format_verdict(document.claim_reports[0].verdict), {"Supported", "Contradicted", "Needs more evidence"})
        self.assertTrue(document.claim_reports[0].profile.family)
        self.assertTrue(document.claim_reports[0].profile.next_step)


if __name__ == "__main__":
    unittest.main()
