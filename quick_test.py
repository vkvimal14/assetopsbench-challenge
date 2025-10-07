"""
Quick test for both example scenarios
Save as: quick_test.py

Usage: python quick_test.py
"""

from main_solution import SupervisorAgent
import json

def test_scenario_113():
    """Test Scenario 113: Sensor Identification"""
    
    print("="*60)
    print("Testing Scenario 113: Sensor Identification")
    print("="*60)
    
    supervisor = SupervisorAgent()
    
    scenario = {
        "id": 113,
        "question": "If Evaporator Water side fouling occurs for Chiller 6, which sensor is most relevant for monitoring this specific failure?"
    }
    
    result = supervisor.solve_scenario(scenario["id"], scenario["question"])
    
    print("\n✅ Result:")
    print(json.dumps(result, indent=2))
    
    # Check if successful
    if result.get('recommended_sensor'):
        print("\n✓ Test PASSED")
        return True
    else:
        print("\n❌ Test FAILED")
        return False

def test_scenario_217():
    """Test Scenario 217: Time Series Forecasting"""
    
    print("\n" + "="*60)
    print("Testing Scenario 217: Time Series Forecasting")
    print("="*60)
    
    supervisor = SupervisorAgent()
    
    scenario = {
        "id": 217,
        "question": "Forecast 'Chiller 9 Condenser Water Flow' using data in 'chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp."
    }
    
    result = supervisor.solve_scenario(scenario["id"], scenario["question"])
    
    print("\n✅ Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Check if successful
    if 'error' not in result and result.get('forecast'):
        print("\n✓ Test PASSED")
        
        # Show first few forecast values
        forecasts = result['forecast'].get('forecasts', [])
        if forecasts:
            print(f"\nFirst 5 forecast values:")
            for i, val in enumerate(forecasts[:5]):
                print(f"  Hour {i+1}: {val:.2f}")
        
        return True
    else:
        print("\n❌ Test FAILED")
        if 'error' in result:
            print(f"Error: {result['error']}")
        return False

def main():
    """Run all tests"""
    
    print("\n" + "🧪"*30)
    print("Quick Test - Example Scenarios")
    print("🧪"*30 + "\n")
    
    results = []
    
    # Test Scenario 113
    results.append(("Scenario 113", test_scenario_113()))
    
    # Test Scenario 217
    results.append(("Scenario 217", test_scenario_217()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("Your setup is working correctly!")
        print("\nNext step: Run process_all_scenarios.py")
    else:
        print("❌ Some tests failed")
        print("Please check the errors above")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()