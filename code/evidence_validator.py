"""Deterministic application of evidence_requirements.csv."""

from __future__ import annotations

from utils import ClaimIntent, EvidenceResult, ImageObservation, normalize_text


def _applicable(issue: str, part: str, text: str) -> bool:
    text = normalize_text(text)
    if "general" in text or "reviewability" in text or "multi-image" in text:
        return True
    issue_terms = {
        "dent": ("dent", "scratch"), "scratch": ("dent", "scratch"),
        "crack": ("crack", "broken", "missing"),
        "glass_shatter": ("crack", "broken", "missing"),
        "broken_part": ("crack", "broken", "missing"),
        "missing_part": ("crack", "broken", "missing", "contents", "inner item"),
        "torn_packaging": ("crushed", "torn", "seal"),
        "crushed_packaging": ("crushed", "torn", "seal"),
        "water_damage": ("water", "stain", "label"),
        "stain": ("water", "stain", "label"),
    }
    return any(term in text for term in issue_terms.get(issue, ())) or part.replace("_", " ") in text


def validate_evidence(
    intent: ClaimIntent,
    observations: list[ImageObservation],
    requirements: list[dict[str, str]],
) -> EvidenceResult:
    matching = [
        row for row in requirements
        if row.get("claim_object") in {"all", intent.claim_object}
        and _applicable(intent.primary_issue, intent.primary_part, row.get("applies_to", ""))
    ]
    requirement_ids = [row["requirement_id"] for row in matching]
    technically_valid = [o for o in observations if o.technical_valid]
    relevant = [
        o for o in technically_valid
        if o.claimed_part_visible and o.claimed_condition_visible and o.confidence >= 0.45
    ]
    if not technically_valid:
        return EvidenceResult(False, False, "No submitted image could be opened for automated review.",
                              [], requirement_ids)
    if not any(not o.error for o in technically_valid):
        return EvidenceResult(False, False, "The submitted images could not be analyzed by the vision model.",
                              [], requirement_ids)
    if not relevant:
        part = intent.primary_part.replace("_", " ")
        contents_claim = (
            intent.claim_object == "package"
            and intent.primary_part in {"contents", "item"}
            and intent.primary_issue == "missing_part"
        )
        visibly_unusable = bool(technically_valid) and all(
            "cropped_or_obstructed" in o.quality_issues for o in technically_valid
        )
        if contents_claim:
            reason = (
                "Missing package contents cannot be established without a usable "
                "view of the opened package and contents area."
            )
        else:
            reason = f"No usable image establishes the claimed {part} condition."
        return EvidenceResult(
            False, not (contents_claim or visibly_unusable), reason,
            [], requirement_ids,
        )
    ids = [o.image_id for o in relevant]
    part = intent.primary_part.replace("_", " ")
    rules_observation = any("deterministic rules" in o.description for o in relevant)
    if rules_observation:
        reason = (
            f"At least one image opened successfully and the extracted {part} claim "
            "is specific enough for deterministic review."
        )
    elif len(ids) == 1:
        reason = f"Image {ids[0]} shows the claimed {part} clearly enough to evaluate the condition."
    else:
        reason = (f"Images {';'.join(ids)} collectively show the claimed {part} clearly enough "
                  "to evaluate the reported condition.")
    return EvidenceResult(True, True, reason, ids, requirement_ids)
