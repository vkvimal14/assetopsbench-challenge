"""
Comprehensive Validation Script for AssetOpsBench Submissions
Validates Track 1 (Planning) and Track 2 (Execution) submissions for compliance and quality.
"""

import os
import zipfile
import json
import sys
from pathlib import Path

def validate_track1():
    """Validate Track 1 submission structure and content."""
    print("\n" + "="*80)
    print("TRACK 1 (PLANNING) VALIDATION")
    print("="*80)
    
    issues = []
    enhancements = []
    
    # Check ZIP exists
    zip_path = Path("submission_track1.zip")
    if not zip_path.exists():
        issues.append("❌ submission_track1.zip not found!")
        return issues, enhancements
    
    print(f"✅ ZIP file exists: {zip_path.name}")
    
    # Check ZIP contents
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            print(f"   Contents: {files}")
            
            required_files = ["track1_planning.py", "track1_fact_sheet.json"]
            for req_file in required_files:
                if req_file not in files:
                    issues.append(f"❌ Missing required file in ZIP: {req_file}")
                else:
                    print(f"   ✅ {req_file} present")
            
            # Validate planning file
            if "track1_planning.py" in files:
                content = zf.read("track1_planning.py").decode('utf-8')
                
                # Check for enhancements
                if "🎯" in content or "Capabilities:" in content:
                    enhancements.append("✨ Enhanced agent descriptions with emojis and structure")
                
                if "CRITICAL CONSTRAINTS" in content or "OUTPUT FORMAT" in content:
                    enhancements.append("✨ Improved planning prompt with constraints and format guidance")
                
                if "PLANNING TIPS" in content:
                    enhancements.append("✨ Added planning tips section")
                
                # Check basic structure
                if "def generate_steps" in content:
                    print("   ✅ generate_steps() method present")
                else:
                    issues.append("❌ Missing generate_steps() method")
                
                if "def get_prompt" in content:
                    print("   ✅ get_prompt() method present")
                else:
                    issues.append("❌ Missing get_prompt() method")
            
            # Validate fact sheet
            if "track1_fact_sheet.json" in files:
                fact_content = zf.read("track1_fact_sheet.json").decode('utf-8')
                try:
                    fact_data = json.loads(fact_content)
                    required_keys = ["task_type", "track", "framework", "model", "description"]
                    for key in required_keys:
                        if key not in fact_data:
                            issues.append(f"❌ Missing key in fact sheet: {key}")
                        else:
                            print(f"   ✅ Fact sheet has '{key}': {fact_data[key]}")
                    
                    # Check model is fixed
                    if fact_data.get("model") != "meta-llama/llama-3-70b-instruct":
                        issues.append("❌ Model must be 'meta-llama/llama-3-70b-instruct'")
                    
                except json.JSONDecodeError:
                    issues.append("❌ Invalid JSON in fact sheet")
    
    except zipfile.BadZipFile:
        issues.append("❌ Invalid ZIP file format")
    except Exception as e:
        issues.append(f"❌ Error reading ZIP: {str(e)}")
    
    return issues, enhancements


def validate_track2():
    """Validate Track 2 submission structure and content."""
    print("\n" + "="*80)
    print("TRACK 2 (EXECUTION) VALIDATION")
    print("="*80)
    
    issues = []
    enhancements = []
    
    # Check ZIP exists
    zip_path = Path("submission_track2.zip")
    if not zip_path.exists():
        issues.append("❌ submission_track2.zip not found!")
        return issues, enhancements
    
    print(f"✅ ZIP file exists: {zip_path.name}")
    
    # Check ZIP contents
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            print(f"   Contents: {files}")
            
            required_files = ["track2_execution.py", "track2_fact_sheet.json"]
            for req_file in required_files:
                if req_file not in files:
                    issues.append(f"❌ Missing required file in ZIP: {req_file}")
                else:
                    print(f"   ✅ {req_file} present")
            
            # Validate execution file
            if "track2_execution.py" in files:
                content = zf.read("track2_execution.py").decode('utf-8')
                
                # Check for enhancements
                if "TaskRevisionHelperAgent" in content:
                    enhancements.append("✨ Implemented TaskRevisionHelperAgent for input refinement")
                
                if "fallback" in content.lower() or "secondary_agent" in content:
                    enhancements.append("✨ Added fallback execution strategy")
                
                if "logging" in content.lower() or "logger" in content:
                    enhancements.append("✨ Enhanced logging for debugging")
                
                if "try:" in content and "except" in content:
                    enhancements.append("✨ Robust error handling implemented")
                
                # Check basic structure
                if "class DynamicWorkflow" in content:
                    print("   ✅ DynamicWorkflow class present")
                else:
                    issues.append("❌ Missing DynamicWorkflow class")
                
                if "def run" in content:
                    print("   ✅ run() method present")
                else:
                    issues.append("❌ Missing run() method")
            
            # Validate fact sheet
            if "track2_fact_sheet.json" in files:
                fact_content = zf.read("track2_fact_sheet.json").decode('utf-8')
                try:
                    fact_data = json.loads(fact_content)
                    required_keys = ["task_type", "track", "framework", "model", "description"]
                    for key in required_keys:
                        if key not in fact_data:
                            issues.append(f"❌ Missing key in fact sheet: {key}")
                        else:
                            print(f"   ✅ Fact sheet has '{key}': {fact_data[key]}")
                    
                    # Check model is fixed
                    if fact_data.get("model") != "meta-llama/llama-3-70b-instruct":
                        issues.append("❌ Model must be 'meta-llama/llama-3-70b-instruct'")
                    
                except json.JSONDecodeError:
                    issues.append("❌ Invalid JSON in fact sheet")
    
    except zipfile.BadZipFile:
        issues.append("❌ Invalid ZIP file format")
    except Exception as e:
        issues.append(f"❌ Error reading ZIP: {str(e)}")
    
    return issues, enhancements


def check_source_files():
    """Verify source files exist in the correct location."""
    print("\n" + "="*80)
    print("SOURCE FILES CHECK")
    print("="*80)
    
    issues = []
    
    source_files = [
        "src/agent_hive/workflows/track1_planning.py",
        "src/agent_hive/workflows/track1_fact_sheet.json",
        "src/agent_hive/workflows/track2_execution.py",
        "src/agent_hive/workflows/track2_fact_sheet.json"
    ]
    
    for file_path in source_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            issues.append(f"❌ Missing source file: {file_path}")
    
    return issues


def main():
    """Run all validations."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "ASSETOPSBENCH SUBMISSION VALIDATOR" + " "*24 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    all_issues = []
    all_enhancements = []
    
    # Check source files
    src_issues = check_source_files()
    all_issues.extend(src_issues)
    
    # Validate Track 1
    t1_issues, t1_enhancements = validate_track1()
    all_issues.extend(t1_issues)
    all_enhancements.extend(t1_enhancements)
    
    # Validate Track 2
    t2_issues, t2_enhancements = validate_track2()
    all_issues.extend(t2_issues)
    all_enhancements.extend(t2_enhancements)
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if all_enhancements:
        print("\n🎉 ENHANCEMENTS DETECTED:")
        for enhancement in all_enhancements:
            print(f"  {enhancement}")
    
    if all_issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in all_issues:
            print(f"  {issue}")
        print("\n❌ VALIDATION FAILED - Please fix the issues above")
        return 1
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("   Your submissions are ready for upload to CodaBench.")
        print("\n📦 Submission Files:")
        print("   • submission_track1.zip")
        print("   • submission_track2.zip")
        print("\n🚀 Next Steps:")
        print("   1. Go to https://www.codabench.org/competitions/4182/")
        print("   2. Navigate to 'My Submissions'")
        print("   3. Upload submission_track1.zip for Track 1")
        print("   4. Upload submission_track2.zip for Track 2")
        print("   5. Wait for evaluation results")
        return 0


if __name__ == "__main__":
    sys.exit(main())
