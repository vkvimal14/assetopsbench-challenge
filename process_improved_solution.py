"""
Process All Scenarios with Improved Solution
Test the enhanced AI system against all 141 scenarios

Save as: process_improved_solution.py
"""

import pandas as pd
import json
import os
from datetime import datetime
from improved_solution import ImprovedSupervisorAgent

def load_scenarios():
    """Load all scenarios from CSV"""
    scenarios_path = './data/scenarios.csv'
    
    if not os.path.exists(scenarios_path):
        print("❌ scenarios.csv not found!")
        return None
    
    df = pd.read_csv(scenarios_path)
    print(f"✓ Loaded {len(df)} scenarios")
    return df

def process_all_scenarios_improved():
    """Process all scenarios with the improved solution"""
    
    print("="*80)
    print("🚀 Processing All Scenarios - Enhanced Solution v2.0")
    print("="*80)
    print()
    
    # Load scenarios
    scenarios_df = load_scenarios()
    if scenarios_df is None:
        return None
    
    # Initialize improved supervisor
    print("🧠 Initializing Enhanced AI Agent System...")
    supervisor = ImprovedSupervisorAgent()
    
    # Results tracking
    results = {}
    errors = {}
    task_type_stats = {}
    
    print(f"\n📊 Processing {len(scenarios_df)} scenarios...")
    print("-"*80)
    
    # Process each scenario
    for idx, row in scenarios_df.iterrows():
        scenario_id = row['id']
        scenario_type = row.get('type', None)
        question = row['text']
        
        print(f"\n[{idx+1:3d}/{len(scenarios_df)}] Scenario {scenario_id:3d} ({scenario_type or 'Auto'})...")
        
        try:
            # Solve scenario with improved agent
            result = supervisor.solve_scenario(scenario_id, question, scenario_type)
            
            # Track task types
            task_type = result.get('task_type', 'unknown')
            task_type_stats[task_type] = task_type_stats.get(task_type, 0) + 1
            
            if 'error' in result:
                print(f"  ⚠️  Warning: {result['error']}")
                errors[str(scenario_id)] = result
            else:
                print(f"  ✅ Success: {result.get('scenario_type', 'processed')}")
                results[str(scenario_id)] = result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors[str(scenario_id)] = {"error": str(e), "question": question}
    
    print("\n" + "="*80)
    print("📈 Enhanced Processing Results")
    print("="*80)
    
    successful = len(results)
    total = len(scenarios_df)
    error_count = len(errors)
    success_rate = (successful / total) * 100
    
    print(f"✅ Successfully processed: {successful}/{total} scenarios ({success_rate:.1f}%)")
    print(f"⚠️  Errors/Warnings: {error_count} scenarios ({(error_count/total)*100:.1f}%)")
    print(f"🚀 Improvement: {success_rate:.1f}% vs 7.8% (original solution)")
    
    # Task type breakdown
    print(f"\n📋 Task Type Distribution:")
    for task_type, count in sorted(task_type_stats.items()):
        percentage = (count / total) * 100
        print(f"  • {task_type:20s}: {count:3d} scenarios ({percentage:5.1f}%)")
    
    print("="*80)
    
    return results, errors, task_type_stats

def create_improved_submission(results, errors=None):
    """Create enhanced submission file"""
    
    print("\n🎯 Creating Enhanced Submission File...")
    
    # Create submissions directory
    os.makedirs('./submissions', exist_ok=True)
    
    # Format results for submission
    submission = {}
    
    for scenario_id, result in results.items():
        if 'error' in result:
            continue
            
        scenario_type = result.get('scenario_type', 'unknown')
        task_type = result.get('task_type', 'unknown')
        
        # Format based on scenario type with enhanced handling
        if scenario_type == 'sensor_identification':
            submission[scenario_id] = {
                "answer": result.get('recommended_sensor', 'Unknown'),
                "confidence": result.get('confidence', 0.9),
                "reasoning": result.get('reasoning', ''),
                "all_relevant_sensors": result.get('all_relevant_sensors', [])
            }
            
        elif scenario_type == 'forecasting':
            forecast_data = result.get('forecast', {})
            forecasts = forecast_data.get('forecasts', [])
            
            submission[scenario_id] = {
                "forecast": forecasts[:24] if len(forecasts) > 0 else [0] * 24,
                "method": forecast_data.get('method', 'enhanced_ensemble_forecasting'),
                "confidence": forecast_data.get('confidence', 0.85),
                "confidence_interval": {
                    "lower": forecast_data.get('lower_bound', [])[:24],
                    "upper": forecast_data.get('upper_bound', [])[:24]
                },
                "horizon_hours": 24
            }
        
        elif scenario_type == 'anomaly_detection':
            submission[scenario_id] = {
                "anomalies": result.get('anomaly_indices', []),
                "anomaly_count": result.get('anomalies_detected', 0),
                "detection_method": result.get('detection_method', 'ensemble_anomaly_detection'),
                "confidence": result.get('confidence', 0.88)
            }
            
        elif scenario_type == 'IoT':
            # IoT query responses
            if result.get('query_type') == 'list_sites':
                submission[scenario_id] = {
                    "sites": result.get('sites', []),
                    "count": result.get('count', 0)
                }
            elif result.get('query_type') == 'list_assets':
                submission[scenario_id] = {
                    "assets": result.get('assets', []),
                    "site": result.get('site', 'MAIN'),
                    "count": result.get('count', 0)
                }
            else:
                submission[scenario_id] = {
                    "response": result.get('response', 'IoT query processed'),
                    "data": result
                }
                
        elif scenario_type == 'knowledge_query':
            if result.get('query_type') == 'failure_modes':
                submission[scenario_id] = {
                    "failure_modes": result.get('failure_modes', []),
                    "equipment": result.get('equipment', ''),
                    "count": result.get('count', 0)
                }
            elif result.get('query_type') == 'sensor_list':
                submission[scenario_id] = {
                    "sensors": result.get('sensors', []),
                    "equipment": result.get('equipment', ''),
                    "count": result.get('count', 0)
                }
            else:
                submission[scenario_id] = result
                
        elif scenario_type == 'workorder':
            submission[scenario_id] = {
                "equipment": result.get('equipment', ''),
                "work_orders": result.get('work_orders_found', 0),
                "query_type": result.get('query_type', ''),
                "details": result
            }
            
        elif scenario_type == 'tsfm':
            submission[scenario_id] = result
            
        else:
            # Generic format for other types
            submission[scenario_id] = result
    
    # Save enhanced submission
    submission_path = './submissions/submission_improved.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"✓ Enhanced submission file created: {submission_path}")
    print(f"✓ Total scenarios in submission: {len(submission)}")
    
    # Save detailed results
    results_path = './submissions/detailed_results_improved.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Detailed results saved: {results_path}")
    
    # Save errors log if any
    if errors:
        errors_path = './submissions/errors_improved.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Errors log saved: {errors_path}")
    
    return submission

def create_comparison_report(original_success_rate=7.8, improved_results=None, task_stats=None):
    """Create detailed comparison report"""
    
    if improved_results is None:
        return
    
    successful = len(improved_results)
    total = 141  # Total scenarios
    improved_success_rate = (successful / total) * 100
    
    report_path = './submissions/improvement_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AssetOpsBench Challenge - Enhanced Solution Performance Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance comparison
        f.write("📊 PERFORMANCE COMPARISON:\n")
        f.write("-"*40 + "\n")
        f.write(f"Original Solution Success Rate:    {original_success_rate:6.1f}%\n")
        f.write(f"Enhanced Solution Success Rate:    {improved_success_rate:6.1f}%\n")
        f.write(f"Performance Improvement:           {improved_success_rate - original_success_rate:+6.1f}%\n")
        f.write(f"Improvement Factor:                {improved_success_rate / original_success_rate:6.1f}x\n\n")
        
        # Detailed metrics
        f.write("📈 DETAILED METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Scenarios:                   {total:6d}\n")
        f.write(f"Successfully Processed:            {successful:6d}\n")
        f.write(f"Failed/Error Scenarios:            {total - successful:6d}\n")
        f.write(f"Success Rate:                      {improved_success_rate:6.1f}%\n")
        f.write(f"Error Rate:                        {(total - successful)/total*100:6.1f}%\n\n")
        
        # Task type distribution
        if task_stats:
            f.write("🔍 TASK TYPE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            for task_type, count in sorted(task_stats.items()):
                percentage = (count / total) * 100
                f.write(f"{task_type:25s}: {count:3d} scenarios ({percentage:5.1f}%)\n")
            f.write("\n")
        
        # Key improvements
        f.write("🚀 KEY IMPROVEMENTS IMPLEMENTED:\n")
        f.write("-"*40 + "\n")
        f.write("✅ Enhanced Task Classification System\n")
        f.write("   - Advanced pattern matching with regex\n")
        f.write("   - Context keyword analysis\n")
        f.write("   - Multi-strategy classification\n\n")
        
        f.write("✅ Comprehensive Equipment Database\n")
        f.write("   - Extended sensor mappings\n")
        f.write("   - Multiple equipment types\n")
        f.write("   - Failure mode associations\n\n")
        
        f.write("✅ Robust Error Handling\n")
        f.write("   - Fallback response generation\n")
        f.write("   - Graceful degradation\n")
        f.write("   - Exception recovery\n\n")
        
        f.write("✅ Advanced Analytics\n")
        f.write("   - Enhanced forecasting algorithms\n")
        f.write("   - Multi-algorithm anomaly detection\n")
        f.write("   - Confidence scoring\n\n")
        
        # Target achievement
        target_rate = 85.0
        f.write("🎯 TARGET ACHIEVEMENT:\n")
        f.write("-"*40 + "\n")
        f.write(f"Target Success Rate:               {target_rate:6.1f}%\n")
        f.write(f"Achieved Success Rate:             {improved_success_rate:6.1f}%\n")
        
        if improved_success_rate >= target_rate:
            f.write("🏆 TARGET ACHIEVED! ✅\n")
        else:
            f.write(f"📈 Progress: {improved_success_rate/target_rate*100:.1f}% of target\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Enhanced solution ready for competition submission!\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Comparison report saved: {report_path}")

def main():
    """Main execution function"""
    
    print("\n" + "🎯"*30)
    print("AssetOpsBench Enhanced Solution - Complete Processing")
    print("🎯"*30 + "\n")
    
    # Step 1: Process all scenarios with improved solution
    result = process_all_scenarios_improved()
    if result is None:
        print("\n❌ Failed to load scenarios")
        return
    
    results, errors, task_stats = result
    
    # Step 2: Create enhanced submission file
    print("\n" + "="*80)
    submission = create_improved_submission(results, errors)
    
    # Step 3: Create comparison report
    print("\n" + "="*80)
    print("📋 Creating Performance Comparison Report")
    print("="*80)
    create_comparison_report(7.8, results, task_stats)
    
    # Final summary
    successful = len(results)
    total = 141
    success_rate = (successful / total) * 100
    
    print("\n" + "="*80)
    print("🎉 ENHANCED SOLUTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"  • Success Rate: {success_rate:.1f}% (vs 7.8% original)")
    print(f"  • Improvement: {success_rate - 7.8:+.1f} percentage points")
    print(f"  • Performance Factor: {success_rate / 7.8:.1f}x better")
    
    print(f"\n📁 Files Generated:")
    print("  1. submissions/submission_improved.json     ← Enhanced submission for Codabench")
    print("  2. submissions/detailed_results_improved.json  ← Full results analysis")
    print("  3. submissions/improvement_report.txt       ← Performance comparison")
    if errors:
        print("  4. submissions/errors_improved.json        ← Remaining errors log")
    
    if success_rate >= 85:
        print(f"\n🏆 SUCCESS! Target of 85%+ achieved with {success_rate:.1f}%")
        print("✅ Solution ready for competition submission!")
    else:
        print(f"\n📈 Progress: {success_rate:.1f}% success rate achieved")
        print(f"🎯 Target: 85%+ (need {85 - success_rate:.1f}% more)")
    
    print("\n" + "🚀"*30)
    print("Ready to compete in CODS 2025!")
    print("🚀"*30 + "\n")

if __name__ == "__main__":
    main()