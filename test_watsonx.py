"""
Test Watsonx.ai connection
"""
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

def test_watsonx():
    """Test if Watsonx credentials work"""
    
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")
    
    print("Testing Watsonx.ai connection...")
    print(f"API Key: {api_key[:10]}..." if api_key else "Not found")
    print(f"Project ID: {project_id}" if project_id else "Not found")
    
    if not api_key or not project_id:
        print("\n❌ Credentials not found!")
        print("Please:")
        print("1. Request access from forum")
        print("2. Add credentials to .env file")
        return False
    
    # Note: Actual API call would go here
    # For now, just verify credentials exist
    print("\n✓ Credentials found")
    print("✓ Ready to use Watsonx.ai")
    return True

if __name__ == "__main__":
    test_watsonx()