"""
Process all scenarios and create submission
Save as: process_all_scenarios.py

Usage: python process_all_scenarios.py
"""

import pandas as pd
import json
import os
from datetime import datetime
from main_solution import SupervisorAgent

def load_scenarios():
    """Load all scenarios from CSV"""
    scenarios_path = './data/scenarios.csv'
    
    if not os.path.exists(scenarios_path):
        print("❌ scenarios.csv not found!")
        print("Run: python download_data.py")
        return None
    
    df = pd.read_csv(scenarios_path)
    print(f"✓ Loaded {len(df)} scenarios")
    return df

def process_all_scenarios():
    """Process all scenarios from the dataset"""
    
    print("="*60)
    print("Processing All Scenarios")
    print("="*60)
    
    # Load scenarios
    scenarios_df = load_scenarios()
    if scenarios_df is None:
        return None
    
    # Initialize supervisor
    print("\nInitializing agent system...")
    supervisor = SupervisorAgent()
    
    # Results dictionary
    results = {}
    errors = {}
    
    print(f"\nProcessing {len(scenarios_df)} scenarios...")
    print("-"*60)
    
    # Process each scenario
    for idx, row in scenarios_df.iterrows():
        scenario_id = row['id']
        scenario_type = row['type']
        question = row['text']
        
        print(f"\n[{idx+1}/{len(scenarios_df)}] Scenario {scenario_id} ({scenario_type})...")
        
        try:
            # Solve scenario
            result = supervisor.solve_scenario(scenario_id, question)
            
            if 'error' in result:
                print(f"  ⚠️  Warning: {result['error']}")
                errors[str(scenario_id)] = result
            else:
                print(f"  ✓ Completed")
                results[str(scenario_id)] = result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors[str(scenario_id)] = {"error": str(e), "question": question}
    
    print("\n" + "="*60)
    print(f"✓ Successfully processed: {len(results)} scenarios")
    print(f"⚠️  Errors/Warnings: {len(errors)} scenarios")
    print("="*60)
    
    return results, errors

def create_submission_file(results, errors=None):
    """Create submission file in the correct format"""
    
    print("\nCreating submission file...")
    
    # Create submissions directory
    os.makedirs('./submissions', exist_ok=True)
    
    # Format results for submission
    submission = {}
    
    for scenario_id, result in results.items():
        if 'error' in result:
            continue
            
        scenario_type = result.get('scenario_type', 'unknown')
        
        # Format based on scenario type
        if scenario_type == 'sensor_identification':
            submission[scenario_id] = {
                "answer": result.get('recommended_sensor', 'Unknown'),
                "confidence": 0.85,
                "reasoning": result.get('reasoning', '')
            }
            
        elif scenario_type == 'forecasting':
            forecast_data = result.get('forecast', {})
            forecasts = forecast_data.get('forecasts', [])
            
            submission[scenario_id] = {
                "forecast": forecasts[:24] if len(forecasts) > 0 else [0] * 24,
                "method": "multi_agent_ensemble",
                "confidence_interval": {
                    "lower": forecast_data.get('lower_bound', [])[:24],
                    "upper": forecast_data.get('upper_bound', [])[:24]
                }
            }
        
        elif scenario_type == 'anomaly_detection':
            submission[scenario_id] = {
                "anomalies": result.get('anomaly_indices', []),
                "anomaly_count": result.get('anomalies_found', 0)
            }
            
        elif scenario_type == 'root_cause_analysis':
            submission[scenario_id] = {
                "analysis": result.get('analysis', ''),
                "equipment": result.get('equipment', ''),
                "recommended_actions": result.get('recommended_actions', [])
            }
        else:
            # Generic format for unknown types
            submission[scenario_id] = result
    
    # Save submission
    submission_path = './submissions/submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"✓ Submission file created: {submission_path}")
    print(f"✓ Total scenarios in submission: {len(submission)}")
    
    # Save detailed results (for debugging)
    results_path = './submissions/detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Detailed results saved: {results_path}")
    
    # Save errors log
    if errors:
        errors_path = './submissions/errors.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Errors log saved: {errors_path}")
    
    return submission

def validate_submission(submission_path='./submissions/submission.json'):
    """Validate submission file"""
    
    print("\n" + "="*60)
    print("Validating Submission")
    print("="*60)
    
    if not os.path.exists(submission_path):
        print("❌ Submission file not found!")
        return False
    
    with open(submission_path, 'r') as f:
        submission = json.load(f)
    
    print(f"✓ Submission file loaded")
    print(f"✓ Number of scenarios: {len(submission)}")
    
    # Check format
    issues = []
    for scenario_id, data in submission.items():
        # Check if scenario ID is numeric string
        try:
            int(scenario_id)
        except ValueError:
            issues.append(f"Invalid scenario ID: {scenario_id}")
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            issues.append(f"Scenario {scenario_id}: data should be a dictionary")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Submission format is valid")
    print("\nSubmission Statistics:")
    print(f"  • Total scenarios: {len(submission)}")
    
    # Count scenario types
    types_count = {}
    for data in submission.values():
        if 'answer' in data:
            types_count['sensor_identification'] = types_count.get('sensor_identification', 0) + 1
        elif 'forecast' in data:
            types_count['forecasting'] = types_count.get('forecasting', 0) + 1
        elif 'anomalies' in data:
            types_count['anomaly_detection'] = types_count.get('anomaly_detection', 0) + 1
        elif 'analysis' in data:
            types_count['root_cause'] = types_count.get('root_cause', 0) + 1
    
    for type_name, count in types_count.items():
        print(f"  • {type_name}: {count}")
    
    return True

def create_summary_report(results, errors):
    """Create a summary report of the processing"""
    
    report_path = './submissions/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AssetOpsBench Challenge - Processing Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Scenarios Processed: {len(results) + len(errors)}\n")
        f.write(f"Successfully Processed: {len(results)}\n")
        f.write(f"Errors/Warnings: {len(errors)}\n\n")
        
        # Scenario type breakdown
        f.write("-"*60 + "\n")
        f.write("Scenario Types:\n")
        f.write("-"*60 + "\n")
        
        types_count = {}
        for result in results.values():
            stype = result.get('scenario_type', 'unknown')
            types_count[stype] = types_count.get(stype, 0) + 1
        
        for stype, count in sorted(types_count.items()):
            f.write(f"  {stype}: {count}\n")
        
        # Error details
        if errors:
            f.write("\n" + "-"*60 + "\n")
            f.write("Errors/Warnings:\n")
            f.write("-"*60 + "\n")
            for scenario_id, error in errors.items():
                f.write(f"\nScenario {scenario_id}:\n")
                f.write(f"  Error: {error.get('error', 'Unknown')}\n")
    
    print(f"✓ Summary report saved: {report_path}")

def main():
    """Main execution"""
    
    print("\n" + "🚀"*30)
    print("AssetOpsBench Challenge - Submission Generator")
    print("🚀"*30 + "\n")
    
    # Step 1: Process all scenarios
    result = process_all_scenarios()
    if result is None:
        print("\n❌ Failed to load scenarios")
        return
    
    results, errors = result
    
    # Step 2: Create submission file
    print("\n" + "="*60)
    submission = create_submission_file(results, errors)
    
    # Step 3: Validate submission
    is_valid = validate_submission()
    
    # Step 4: Create summary report
    print("\n" + "="*60)
    print("Creating Summary Report")
    print("="*60)
    create_summary_report(results, errors)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ SUBMISSION READY!")
    print("="*60)
    print("\n📁 Files created:")
    print("  1. submissions/submission.json        ← Upload this to Codabench")
    print("  2. submissions/detailed_results.json  ← Full results for review")
    print("  3. submissions/errors.json            ← Error log")
    print("  4. submissions/summary_report.txt     ← Summary report")
    
    if is_valid:
        print("\n✓ Submission is valid and ready to upload!")
        print("\n📤 Next steps:")
        print("  1. Review submissions/summary_report.txt")
        print("  2. Check submissions/errors.json if any")
        print("  3. Upload submissions/submission.json to Codabench")
        print("  4. Wait for evaluation results")
    else:
        print("\n⚠️  Please fix validation errors before uploading")
    
    print("\n" + "🎯"*30)
    print("Good Luck!")
    print("🎯"*30 + "\n")

if __name__ == "__main__":
    main()