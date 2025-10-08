#!/usr/bin/env python3
"""Test Wind Turbine scenarios specifically"""

import pandas as pd
from main_solution import LLMAssetOpsBenchProcessor

def test_wind_turbine_scenarios():
    """Test specific Wind Turbine scenarios"""
    
    # Load scenarios
    scenarios_df = pd.read_csv("data/scenarios.csv")
    
    # Target scenarios
    target_scenarios = [103, 105]
    
    # Initialize solution
    solution = LLMAssetOpsBenchProcessor()
    
    for scenario_id in target_scenarios:
        scenario = scenarios_df[scenarios_df['id'] == scenario_id]
        if not scenario.empty:
            query = scenario.iloc[0]['text']
            print(f"\n{'='*60}")
            print(f"Scenario {scenario_id}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            try:
                # Process the query
                result = solution.supervisor.process_query(query, scenario_id)
                print(f"✅ Success: {result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print(f"Scenario {scenario_id} not found")

if __name__ == "__main__":
    test_wind_turbine_scenarios()