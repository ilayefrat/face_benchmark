task_groups = {
    "LFW": ["LFW: Accuracy"],
    "Inversion Effect": [
        "Inversion Effect - Upright: Accuracy",
        "Inversion Effect - Inverted: Accuracy",
    ],
    "Other Race Effect": [
        "Other Race Effect - Caucasian: Accuracy",
        "Other Race Effect - Asian: Accuracy"
    ],
    "View Invariant": [
        "View Invariant - Frontal - Frontal - Same: Mean",
        "View Invariant - Frontal - Quarter Left: Mean",
        "View Invariant - Frontal - Half Left: Mean",
        "View Invariant - Frontal - Frontal - Diff: Mean",
    ],
    "Critical Features": [
        "Critical Features - same: Mean",
        "Critical Features - non_critical_changes: Mean",
        "Critical Features - critical_changes: Mean",
        "Critical Features - diff: Mean"  
    ],
    "IL Celebs": [
        "IL Celebs - Familiar Performance: Correlation Score",
        "IL Celebs - Unfamiliar Performance: Correlation Score"
    ],
    "International Celebs": [
        "International Celebs - Visual Perception Similarity: Correlation Score",
        "International Celebs - Memory Perception Similarity: Correlation Score"
    ],
    # ... Add more task groups and selected metrics
}

pretty_task_names = {
    "LFW: Accuracy": "Accuracy",
    "LFW: Optimal Threshold": "Optimal Threshold",
    "LFW: AUC": "AUC",
    "Inversion Effect - Upright: Accuracy": "Upright",
    "Inversion Effect - Inverted: Accuracy": "Inverted",
    "Other Race Effect - Caucasian: Accuracy": "Caucasian",
    "Other Race Effect - Asian: Accuracy": "Asian",
    "View Invariant - Frontal - Frontal - Diff: Mean" : "different",
    "View Invariant - Frontal - Frontal - Same: Mean": "same",
    "View Invariant - Frontal - Half Left: Mean":"Half Left",
    "View Invariant - Frontal - Quarter Left: Mean":"Quarter Left",
    "Critical Features - Non-Critical Distances: Correlation Score":"Correlation Score",
    "Critical Features - critical_changes: Mean": "critical",
    "Critical Features - diff: Mean": "diff",
    "Critical Features - non_critical_changes: Mean":"non critical",
    "Critical Features - same: Mean":"same",
    "IL Celebs - Familiar Performance: Correlation Score":"Familiar",
    "IL Celebs - Unfamiliar Performance: Correlation Score":"Unfamiliar",
    "International Celebs - Memory Perception Similarity: Correlation Score":"Memory",
    "International Celebs - Visual Perception Similarity: Correlation Score":"Visual",
    "Inversion Effect - Inverted: AUC":"",

    "Inversion Effect - Inverted: Optimal Threshold":"Inverted",
    "Inversion Effect - Upright: AUC":"Upright: AUC",

    "Inversion Effect - Upright: Optimal Threshold":"Upright",
    "Other Race Effect - Asian: AUC":"Asian: AUC",
    "Other Race Effect - Asian: Optimal Threshold":"Asian",
    "Other Race Effect - Caucasian: AUC":"Caucasian: AUC",
    "Other Race Effect - Caucasian: Optimal Threshold":"Caucasian",
    '"Thatcher Effect: Group ""Inverted"" Mean"':"Inverted",
    '"Thatcher Effect: Group ""Upright"" Mean"':"Upright",
    "Thatcher Effect: Relative Difference":"Relative Difference",

    # Add more mappings as needed...
}