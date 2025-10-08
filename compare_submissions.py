import json

def compare_submissions(ground_truth_file, enhanced_file):
    """Compares the enhanced submission to the ground truth and lists failed scenarios."""
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ground_truth_file}")
        return

    try:
        with open(enhanced_file, 'r') as f:
            enhanced = json.load(f)
    except FileNotFoundError:
        print(f"Error: Enhanced submission file not found at {enhanced_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {enhanced_file}")
        return

    scenarios = ground_truth.keys()
    failed_scenarios = []

    for scenario_id in scenarios:
        if scenario_id not in enhanced:
            failed_scenarios.append(scenario_id)

    print("Failed scenarios:")
    for scenario_id in failed_scenarios:
        print(scenario_id)

if __name__ == "__main__":
    compare_submissions(
        'submissions/submission.json',
        'submissions/submission_llm_enhanced_20251008_073251.json'
    )
