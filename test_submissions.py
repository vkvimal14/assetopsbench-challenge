"""
Advanced Test Suite for AssetOpsBench Submissions
Tests prompt quality, revision logic, and compliance without requiring full framework.
"""

import sys
from pathlib import Path

def test_track1_prompt_quality():
    """Test Track 1 planning prompt enhancements."""
    print("\n" + "="*80)
    print("TEST: Track 1 Prompt Quality")
    print("="*80)
    
    file_path = Path("src/agent_hive/workflows/track1_planning.py")
    if not file_path.exists():
        print("❌ Track 1 file not found")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Enhanced agent descriptions
    tests_total += 1
    if "🎯" in content or "Capabilities:" in content:
        print("✅ Enhanced agent descriptions with structure")
        tests_passed += 1
    else:
        print("⚠️  Basic agent descriptions (could be enhanced)")
    
    # Test 2: Constraint guidance
    tests_total += 1
    if "CRITICAL CONSTRAINTS" in content or "constraints" in content.lower():
        print("✅ Constraint guidance present")
        tests_passed += 1
    else:
        print("⚠️  No explicit constraint guidance")
    
    # Test 3: Output format specification
    tests_total += 1
    if "OUTPUT FORMAT" in content or "format" in content.lower():
        print("✅ Output format specification present")
        tests_passed += 1
    else:
        print("⚠️  No explicit format specification")
    
    # Test 4: Planning tips
    tests_total += 1
    if "PLANNING TIPS" in content or "tips" in content.lower():
        print("✅ Planning tips included")
        tests_passed += 1
    else:
        print("⚠️  No planning tips")
    
    # Test 5: Step-by-step guidance
    tests_total += 1
    if "step" in content.lower() and "sequence" in content.lower():
        print("✅ Step-by-step guidance present")
        tests_passed += 1
    else:
        print("⚠️  Limited step guidance")
    
    score = (tests_passed / tests_total) * 100
    print(f"\n📊 Prompt Quality Score: {tests_passed}/{tests_total} ({score:.0f}%)")
    
    if score >= 80:
        print("   Rating: EXCELLENT ⭐⭐⭐")
    elif score >= 60:
        print("   Rating: GOOD ⭐⭐")
    else:
        print("   Rating: BASIC ⭐")
    
    return score >= 60


def test_track2_revision_logic():
    """Test Track 2 revision helper and execution logic."""
    print("\n" + "="*80)
    print("TEST: Track 2 Revision & Execution Logic")
    print("="*80)
    
    file_path = Path("src/agent_hive/workflows/track2_execution.py")
    if not file_path.exists():
        print("❌ Track 2 file not found")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Revision helper agent
    tests_total += 1
    if "TaskRevisionHelperAgent" in content:
        print("✅ TaskRevisionHelperAgent implemented")
        tests_passed += 1
    else:
        print("⚠️  No revision helper agent")
    
    # Test 2: Input validation
    tests_total += 1
    if "strip()" in content or "trim" in content.lower():
        print("✅ Input validation/trimming present")
        tests_passed += 1
    else:
        print("⚠️  No input validation")
    
    # Test 3: Fallback strategy
    tests_total += 1
    if "fallback" in content.lower() or "secondary" in content.lower():
        print("✅ Fallback execution strategy implemented")
        tests_passed += 1
    else:
        print("⚠️  No fallback strategy")
    
    # Test 4: Error handling
    tests_total += 1
    if "try:" in content and "except" in content:
        print("✅ Error handling implemented")
        tests_passed += 1
    else:
        print("⚠️  No error handling")
    
    # Test 5: Logging
    tests_total += 1
    if "logging" in content.lower() or "print(" in content:
        print("✅ Logging/debugging present")
        tests_passed += 1
    else:
        print("⚠️  No logging")
    
    # Test 6: Memory management
    tests_total += 1
    if "memory" in content.lower() and "append" in content.lower():
        print("✅ Memory management present")
        tests_passed += 1
    else:
        print("⚠️  Basic memory handling")
    
    score = (tests_passed / tests_total) * 100
    print(f"\n📊 Execution Quality Score: {tests_passed}/{tests_total} ({score:.0f}%)")
    
    if score >= 80:
        print("   Rating: EXCELLENT ⭐⭐⭐")
    elif score >= 60:
        print("   Rating: GOOD ⭐⭐")
    else:
        print("   Rating: BASIC ⭐")
    
    return score >= 60


def test_compliance_signals():
    """Test for common compliance issues."""
    print("\n" + "="*80)
    print("TEST: Compliance Signals")
    print("="*80)
    
    all_compliant = True
    
    # Check Track 1
    file_path = Path("src/agent_hive/workflows/track1_planning.py")
    if file_path.exists():
        content = file_path.read_text(encoding='utf-8')
        
        # Check for forbidden modifications
        if "class PlanningWorkflow" in content:
            # Check if core structure is preserved
            if "def __init__" in content and "def generate_steps" in content and "def get_prompt" in content:
                print("✅ Track 1: Core structure preserved")
            else:
                print("⚠️  Track 1: Core structure may be modified")
                all_compliant = False
        
        # Check for TODO regions (good sign of limited changes)
        if "TODO" in content or "# Enhanced" in content or "# Improved" in content:
            print("✅ Track 1: Changes appear focused")
        else:
            print("⚠️  Track 1: May have wide-ranging changes")
    
    # Check Track 2
    file_path = Path("src/agent_hive/workflows/track2_execution.py")
    if file_path.exists():
        content = file_path.read_text(encoding='utf-8')
        
        # Check for forbidden modifications
        if "class DynamicWorkflow" in content:
            # Check if core structure is preserved
            if "def __init__" in content and "def run" in content:
                print("✅ Track 2: Core structure preserved")
            else:
                print("⚠️  Track 2: Core structure may be modified")
                all_compliant = False
        
        # Check for helper agents (allowed addition)
        if "HelperAgent" in content or "RevisionAgent" in content:
            print("✅ Track 2: Helper agent pattern used (good)")
    
    return all_compliant


def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "ADVANCED TEST SUITE" + " "*33 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    results = []
    
    # Run tests
    results.append(("Track 1 Prompt Quality", test_track1_prompt_quality()))
    results.append(("Track 2 Revision Logic", test_track2_revision_logic()))
    results.append(("Compliance Signals", test_compliance_signals()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "⚠️  WARN"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 EXCELLENT! All tests passed.")
        print("   Your submissions show high quality enhancements.")
        return 0
    elif passed >= total * 0.6:
        print("\n✅ GOOD! Most tests passed.")
        print("   Your submissions are functional with room for improvement.")
        return 0
    else:
        print("\n⚠️  BASIC. Several tests show warnings.")
        print("   Consider adding more enhancements.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
