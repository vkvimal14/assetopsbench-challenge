import json
import pandas as pd
import argparse
import os

def analyze_llm_failures(submission_file: str, scenarios_file: str, ground_truth_file: str):
    """
    Analyzes the output of the LLM-enhanced solution to identify and categorize
    failing scenarios. A scenario is considered a failure if its output contains an 'error' key.
    """
    try:
        with open(submission_file, 'r') as f:
            submission_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Submission file not found at '{submission_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{submission_file}'")
        return

    scenarios_df = pd.read_csv(scenarios_file)
    scenarios_df['id'] = scenarios_df['id'].astype(str)

    failing_scenarios = []
    for scenario_id, result in submission_data.items():
        if isinstance(result, dict) and 'error' in result:
            failing_scenarios.append({
                'id': scenario_id,
                'error': result.get('error'),
                'query': result.get('query', 'N/A')
            })

    print(f"\n--- Failure Analysis for {os.path.basename(submission_file)} ---")
    
    if not failing_scenarios:
        print("✅ No scenarios with explicit 'error' keys found. The solution is robust!")
        # Optional: Compare with ground truth for content validation if needed
        # This part can be expanded later.
        return

    print(f"Found {len(failing_scenarios)} failing scenarios with explicit errors:")
    
    fail_df = pd.DataFrame(failing_scenarios)
    
    # Merge with original scenarios to get the full query text if it was missing
    full_fail_info = pd.merge(fail_df, scenarios_df, on='id', how='left')

    for _, row in full_fail_info.iterrows():
        query = row['text'] if pd.notna(row['text']) else row['query']
        print(f"\nID: {row['id']}")
        print(f"  Query: {query}")
        print(f"  Error: {row['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze failing scenarios from the LLM solution output.')
    parser.add_argument('submission_file', help='The JSON submission file to analyze.')
    parser.add_argument('--scenarios_file', default='data/scenarios.csv', help='The CSV file containing all scenario definitions.')
    parser.add_argument('--ground_truth_file', default='submissions/submission.json', help='The ground truth JSON file (for future comparison).')
    
    args = parser.parse_args()

    analyze_llm_failures(args.submission_file, args.scenarios_file, args.ground_truth_file)
