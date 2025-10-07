"""
Test if everything is installed correctly
"""
import sys

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK")
        return True
    else:
        print("❌ Python version too old. Need 3.8+")
        return False

def test_packages():
    """Test if all required packages are installed"""
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'datasets',
        'matplotlib',
        'requests'
    ]
    
    print("\nTesting packages:")
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            all_ok = False
    
    return all_ok

def test_data_files():
    """Check if data files exist"""
    import os
    
    print("\nChecking data files:")
    
    if os.path.exists("./data/scenarios.csv"):
        print("✓ scenarios.csv found")
        return True
    else:
        print("❌ scenarios.csv not found")
        print("Run: python download_data.py")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Setup Verification")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Packages", test_packages),
        ("Data Files", test_data_files)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-"*60)
        result = test_func()
        results.append(result)
    
    print("\n" + "="*60)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("You're ready to start coding!")
    else:
        print("❌ Some tests failed")
        print("Please fix the issues above")
    print("="*60)

if __name__ == "__main__":
    main()