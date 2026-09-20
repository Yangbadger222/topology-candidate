"""Independent labels and priority-ordered repair decisions."""
VALID = {"REAL", "FALSE", "UNCERTAIN"}
CONNECT = {"CONNECT", "NO_CONNECT", "UNCERTAIN"}

def derive_action(component_validity: str, connection_label: str) -> str:
    if component_validity not in VALID or connection_label not in CONNECT: raise ValueError("invalid annotation label")
    if component_validity == "UNCERTAIN" or connection_label == "UNCERTAIN": return "REVIEW"
    if component_validity == "FALSE": return "DELETE_COMPONENT"
    return "ADD_CONNECTION" if connection_label == "CONNECT" else "KEEP"

def make_record(candidate, component_validity="UNCERTAIN", connection_label="UNCERTAIN", notes="", annotator="human", timestamp=None):
    from datetime import datetime, timezone
    return {**candidate.to_dict(), "component_validity": component_validity, "connection_label": connection_label,
            "derived_action": derive_action(component_validity, connection_label), "notes": notes,
            "annotator": annotator, "timestamp": timestamp or datetime.now(timezone.utc).isoformat()}
