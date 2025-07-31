# colors_config.py

colors = {
    "lfw": "black",
    "inversion_upright": "black",
    "lfw_accuracy": "#4daf4a",
    "lfw_auc": "#377eb8",
    "lfw_optimal_treshhold": "green",
    "inversion_inverted": "white",
    "inversion_diff": "green",
    "other_race_caucasian": "black",
    "other_race_asian": "gray",
    "other_race_diff": "green",
    "intl_perception": "pink",
    "intl_memory": "teal",
    "il_familiar": "pink",
    "il_unfamiliar": "olive",
    "critical_same": "#0000FF",
    "critical_noncritical": "#800080",
    "critical_critical": "#006400",
    "critical_diff": "#FF0000",
    "view_same": "#00008B",
    "view_qleft": "#0077FF",
    "view_hleft": "#ADD8E6",
    "view_diff": "#FF0000"
}
col_to_name = {
    "LFW: Accuracy": "lfw",
    "Inversion Effect - Upright: Accuracy":"inversion_upright",
    "Inversion Effect - Inverted: Accuracy":"inversion_inverted",

    "Other Race Effect - Caucasian: Accuracy":"other_race_caucasian",
    "Other Race Effect - Asian: Accuracy":"other_race_asian",

    "View Invariant - Frontal - Frontal - Same: Mean":"view_same",
    "View Invariant - Frontal - Quarter Left: Mean":"view_qleft",
    "View Invariant - Frontal - Half Left: Mean":"view_hleft",
    "View Invariant - Frontal - Frontal - Diff: Mean": "view_diff",


    "Critical Features - same: Mean":"critical_same",
    "Critical Features - non_critical_changes: Mean":"critical_noncritical",
    "Critical Features - critical_changes: Mean":"critical_critical",
    "Critical Features - diff: Mean":"critical_diff",

    "IL Celebs - Familiar Performance: Correlation Score":"il_familiar",
    "IL Celebs - Unfamiliar Performance: Correlation Score":"il_unfamiliar",


    "International Celebs - Visual Perception Similarity: Correlation Score":"intl_perception",
    "International Celebs - Memory Perception Similarity: Correlation Score":"intl_memory",

    # ... Add more task groups and selected metrics
}

dark_variants = {
    "red": "darkred",
    "blue": "darkblue",
    "lightblue": "navy",
    "green": "darkgreen",
    "orange": "#cc5500",
    "purple": "#2e003e",
    "gray": "#2f2f2f",
    "cyan": "#007070",
    "yellow": "#1f1d00",
    "pink": "#cc3366",
    "brown": "saddlebrown",
    "teal": "#006666",
    "olive": "darkolivegreen",
    "moccasin": "indigo",    
    "black": "black",
    "white": "#cccccc",         # Soft gray instead of true white for dark mode
    "#0000FF": "#00008B",       # Blue → DarkBlue
    "#800080": "#2e003e",       # Purple → same dark purple
    "#006400": "#003200",       # DarkGreen → even darker green
    "#FF0000": "darkred",       # Red → DarkRed
    "#00008B": "#000060",       # DarkBlue → even darker
    "#0077FF": "#004488",       # Mid blue → darker tone
    "#ADD8E6": "#5f9ea0"        # LightBlue → CadetBlue
}