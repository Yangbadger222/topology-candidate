FEATURE_NAMES = ["normalized_gap", "gap_distance", "component_nodes", "component_length_px", "source_degree",
                 "target_component_nodes", "target_is_main", "gap_graph_support", "gap_heat_support"]

def feature_matrix(candidates, names=FEATURE_NAMES):
    import numpy as np
    return np.asarray([[float(c.features.get(n, 0.0)) for n in names] for c in candidates], dtype=float)
