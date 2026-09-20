"""Pure annotation-state helpers shared by UI, schema persistence, and QA."""
import re

_CANDIDATE_RE = re.compile(r"(?:^|_)candidate_(\d+)$")

def candidate_rank_from_id(selection):
    if not isinstance(selection, str):
        return None
    match = _CANDIDATE_RE.search(selection)
    return int(match.group(1)) if match else None

def is_candidate_selection(selection):
    return candidate_rank_from_id(selection) is not None or bool(
        isinstance(selection, str) and _CANDIDATE_RE.search(selection)
    )

def is_case_completed(case, annotation):
    """Whether this page is complete; source records may have unfinished endpoints."""
    if not annotation:
        return False
    validity = annotation.get("component_validity")
    if validity in {"FALSE", "UNCERTAIN"}:
        return True
    if case.get("source_endpoint_id") is None:
        return validity == "REAL"
    if validity != "REAL":
        return False
    return str(case["source_endpoint_id"]) in (annotation.get("endpoint_labels") or {})
