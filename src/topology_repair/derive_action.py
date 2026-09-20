"""Priority-ordered V2 repair actions."""
VALID={"REAL","FALSE","UNCERTAIN"}
SELECTIONS={"NO_CONNECTION","CORRECT_TARGET_NOT_PROPOSED","UNCERTAIN"}
from .annotation_logic import is_candidate_selection

def derive_source_action(component_validity):
    if component_validity == "FALSE": return "DELETE_SOURCE_SUBGRAPH"
    if component_validity == "UNCERTAIN": return "REVIEW"
    if component_validity == "REAL": return "KEEP"
    raise ValueError("invalid component validity")

def derive_action(component_validity, connection_label=None, selection=None):
    """Derive an action; FALSE always wins over endpoint uncertainty.

    ``connection_label`` is retained as a compatibility alias for old callers.
    """
    if component_validity not in VALID: raise ValueError("invalid component validity")
    choice=selection if selection is not None else connection_label
    if component_validity == "FALSE": return "DELETE_SOURCE_SUBGRAPH"
    if component_validity == "UNCERTAIN": return "REVIEW"
    if choice in (None,"UNCERTAIN"): return "REVIEW"
    if choice == "CORRECT_TARGET_NOT_PROPOSED": return "CANDIDATE_MISS"
    if choice == "NO_CONNECTION": return "KEEP"
    if is_candidate_selection(choice) or (isinstance(choice, str) and choice.startswith("T")) or choice == "CONNECT": return "ADD_CONNECTION"
    raise ValueError("invalid endpoint selection")

def make_record(candidate, component_validity="UNCERTAIN", connection_label="UNCERTAIN", notes="", annotator="human", timestamp=None):
    from datetime import datetime, timezone
    return {**candidate.to_dict(),"component_validity":component_validity,"connection_label":connection_label,
            "derived_action":derive_action(component_validity,connection_label),"notes":notes,"annotator":annotator,
            "timestamp":timestamp or datetime.now(timezone.utc).isoformat()}
