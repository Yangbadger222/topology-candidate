"""Import the 20-case seed convention from the research spec."""
def seed_label(case_number):
    n=int(case_number)
    return {"component_validity":"FALSE" if n in (6,7,8) else "REAL",
            "connection_label":"CONNECT" if n in (2,3,4,5,6,14) else "NO_CONNECT"}
