"""
Process all scenarios using enhanced solution
Save as: process_with_enhanced.py

Usage: python process_with_enhanced.py
"""

import pandas as pd
import json
import os
from datetime import datetime
from enhanced_solution import EnhancedSupervisorAgent

def process_all_scenarios_enhanced():
    """Process all scenarios with enhanced agent"""
    
    print("="*60)
    print("Processing All Scenarios (Enhanced)")
    print("="*60)
    
    # Load scenarios
    scenarios_path = './data/scenarios.csv'
    if not os.path.exists(scenarios_path):
        print("❌ scenarios.csv not found!")
        return None, None
    
    df = pd.read_csv(scenarios_path)
    print(f"✓ Loaded {len(df)} scenarios")
    
    # Initialize enhanced supervisor
    print("\nInitializing enhanced agent system...")
    supervisor = EnhancedSupervisorAgent()
    
    # Results
    results = {}
    errors = {}
    
    print(f"\nProcessing {len(df)} scenarios...")
    print("-"*60)
    
    # Process each scenario
    for idx, row in df.iterrows():
        scenario_id = row['id']
        scenario_type = row['type'] if pd.notna(row['type']) else None
        question = row['text']
        
        print(f"\n[{idx+1}/{len(df)}] Scenario {scenario_id} ({scenario_type})...")
        
        try:
            # Solve scenario
            result = supervisor.solve_scenario(scenario_id, question, scenario_type)
            
            if 'error' in result and result['scenario_type'] != 'unknown':
                print(f"  ⚠️  Warning: {result['error']}")
                errors[str(scenario_id)] = result
            else:
                if result['scenario_type'] == 'unknown':
                    print(f"  ⚠️  Unknown task type (providing baseline response)")
                else:
                    print(f"  ✓ Completed")
                results[str(scenario_id)] = result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors[str(scenario_id)] = {"error": str(e), "question": question}
    
    print("\n" + "="*60)
    print(f"✓ Successfully processed: {len(results)} scenarios")
    print(f"⚠️  Errors: {len(errors)} scenarios")
    print("="*60)
    
    return results, errors

def create_enhanced_submission(results, errors=None):
    """Create submission with enhanced formatting"""
    
    print("\nCreating enhanced submission file...")
    
    os.makedirs('./submissions', exist_ok=True)
    
    submission = {}
    
    for scenario_id, result in results.items():
        scenario_type = result.get('scenario_type', 'unknown')
        
        # Format based on type
        if scenario_type == 'sensor_identification':
            submission[scenario_id] = {
                "answer": result.get('recommended_sensor', 'Unknown'),
                "confidence": 0.85
            }
            
        elif scenario_type == 'forecasting':
            forecast_data = result.get('forecast', {})
            submission[scenario_id] = {
                "forecast": forecast_data.get('forecasts', [0]*24)[:24],
                "method": forecast_data.get('method', 'baseline')
            }
            
        elif scenario_type == 'IoT':
            query_type = result.get('query_type', 'general')
            if query_type == 'list_sites':
                submission[scenario_id] = {
                    "sites": result.get('sites', []),
                    "count": result.get('count', 0)
                }
            elif query_type == 'list_assets':
                submission[scenario_id] = {
                    "assets": result.get('assets', []),
                    "site": result.get('site', ''),
                    "count": result.get('count', 0)
                }
            else:
                submission[scenario_id] = result
                
        elif scenario_type == 'failure_mode_list':
            submission[scenario_id] = {
                "failure_modes": result.get('failure_modes', []),
                "equipment": result.get('equipment', ''),
                "count": result.get('count', 0)
            }
            
        elif scenario_type == 'sensor_list':
            submission[scenario_id] = {
                "sensors": result.get('sensors', []),
                "equipment": result.get('equipment', ''),
                "count": result.get('count', 0)
            }
            
        elif scenario_type == 'anomaly_detection':
            submission[scenario_id] = {
                "anomalies": result.get('anomaly_indices', []),
                "status": result.get('status', 'No anomalies'),
                "count": result.get('anomalies_found', 0)
            }
            
        elif scenario_type == 'workorder':
            submission[scenario_id] = {
                "work_orders": result.get('work_orders', []),
                "action": result.get('action', 'general'),
                "equipment": result.get('equipment', '')
            }
            
        elif scenario_type == 'root_cause_analysis':
            submission[scenario_id] = {
                "analysis": result.get('analysis', ''),
                "recommended_actions": result.get('recommended_actions', [])
            }
            
        else:  # unknown
            submission[scenario_id] = {
                "note": result.get('note', 'Requires additional data'),
                "type": "unknown"
            }
    
    # Save submission
    submission_path = './submissions/submission_enhanced.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"✓ Enhanced submission created: {submission_path}")
    print(f"✓ Total scenarios: {len(submission)}")
    
    # Save detailed results
    detailed_path = './submissions/detailed_results_enhanced.json'
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Detailed results saved: {detailed_path}")
    
    # Save errors if any
    if errors:
        errors_path = './submissions/errors_enhanced.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Errors saved: {errors_path}")
    
    return submission

def create_enhanced_summary(results, errors):
    """Create enhanced summary report"""
    
    report_path = './submissions/summary_report_enhanced.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AssetOpsBench Challenge - Enhanced Processing Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Scenarios: {len(results) + len(errors)}\n")
        f.write(f"Successfully Processed: {len(results)}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        
        # Breakdown by type
        f.write("-"*60 + "\n")
        f.write("Scenario Types Breakdown:\n")
        f.write("-"*60 + "\n")
        
        type_counts = {}
        for result in results.values():
            stype = result.get('scenario_type', 'unknown')
            type_counts[stype] = type_counts.get(stype, 0) + 1
        
        for stype in sorted(type_counts.keys()):
            f.write(f"  {stype}: {type_counts[stype]}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Summary Complete\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Summary report saved: {report_path}")

def main():
    """Main execution"""
    
    print("\n" + "🚀"*30)
    print("AssetOpsBench Challenge - Enhanced Solution")
    print("🚀"*30 + "\n")
    
    # Process all scenarios
    results, errors = process_all_scenarios_enhanced()
    
    if results is None:
        print("\n❌ Failed to process scenarios")
        return
    
    # Create submission
    print("\n" + "="*60)
    submission = create_enhanced_submission(results, errors)
    
    # Create summary
    print("\n" + "="*60)
    print("Creating Summary Report")
    print("="*60)
    create_enhanced_summary(results, errors)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ENHANCED SUBMISSION READY!")
    print("="*60)
    print("\n📁 Files created:")
    print("  1. submissions/submission_enhanced.json")
    print("  2. submissions/detailed_results_enhanced.json")
    print("  3. submissions/errors_enhanced.json")
    print("  4. submissions/summary_report_enhanced.txt")
    
    print("\n📊 Results Summary:")
    print(f"  • Total scenarios processed: {len(results)}")
    print(f"  • Errors: {len(errors)}")
    print(f"  • Success rate: {len(results)/(len(results)+len(errors))*100:.1f}%")
    
    print("\n📤 Next steps:")
    print("  1. Review summary_report_enhanced.txt")
    print("  2. Upload submission_enhanced.json to Codabench")
    print("  3. Monitor evaluation results")
    print("  4. Iterate based on feedback")
    
    print("\n" + "🎯"*30)
    print("Good Luck!")
    print("🎯"*30 + "\n")

if __name__ == "__main__":
    main()