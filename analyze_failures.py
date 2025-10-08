import json
import pandas as pd
import argparse

def find_failed_scenarios(gt_file, sub_file, scenarios_file):
    """Finds and categorizes failed scenarios by comparing submission with ground truth."""
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    with open(sub_file, 'r') as f:
        submission = json.load(f)
    
    scenarios_df = pd.read_csv(scenarios_file)
    scenarios_df['id'] = scenarios_df['id'].astype(str)
    
    failed_ids = []
    for scenario_id, gt_output in ground_truth.items():
        if scenario_id not in submission or submission[scenario_id] != gt_output:
            failed_ids.append(scenario_id)
            
    if not failed_ids:
        print("No failing scenarios found by direct comparison.")
        # Fallback for when direct comparison is insufficient
        all_ids = set(scenarios_df['id'].unique())
        successful_ids = set(submission.keys())
        failed_ids = list(all_ids - successful_ids)

    print(f"Found {len(failed_ids)} failing scenarios.")
    
    if failed_ids:
        failed_scenarios = scenarios_df[scenarios_df['id'].isin(failed_ids)]
        
        print("\n--- Failing Scenarios ---")
        for index, row in failed_scenarios.iterrows():
            print(f"ID: {row['id']}, Query: {row['text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find failed scenarios by comparing submission with ground truth.')
    parser.add_argument('submission_file', help='The submission file to check.')
    parser.add_argument('--gt_file', default='submissions/submission.json', help='The ground truth submission file.')
    parser.add_argument('--scenarios_file', default='data/scenarios.csv', help='The file with the scenario queries.')
    
    args = parser.parse_args()

    find_failed_scenarios(
        args.gt_file,
        args.submission_file,
        args.scenarios_file
    )
