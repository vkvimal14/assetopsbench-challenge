"""
Script to download AssetOpsBench datasets
"""
from datasets import load_dataset
import os
import pandas as pd

def download_scenarios():
    """Download scenarios dataset"""
    print("Downloading scenarios dataset...")
    
    try:
        # Load dataset from Hugging Face
        ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
        
        # Create data directory
        os.makedirs("./data", exist_ok=True)
        
        # Save as CSV
        df = ds['train'].to_pandas()
        df.to_csv("./data/scenarios.csv", index=False)
        
        print(f"✓ Scenarios downloaded: {len(df)} scenarios")
        print(f"✓ Saved to: ./data/scenarios.csv")
        
        # Show first few scenarios
        print("\nFirst 3 scenarios:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"Error downloading scenarios: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in to Hugging Face")
        print("2. Check your internet connection")
        print("3. Verify dataset name is correct")
        return None

def download_time_series_data():
    """Download time series datasets"""
    print("\nDownloading time series data...")
    
    # Note: Check the dataset page for available time series data
    # https://huggingface.co/datasets/ibm-research/AssetOpsBench
    
    try:
        # Example: If there's a 'timeseries' split
        # ds = load_dataset("ibm-research/AssetOpsBench", "timeseries")
        # For now, we'll create sample data
        
        print("Note: Time series data will be provided separately")
        print("Check the competition page for data files")
        
    except Exception as e:
        print(f"Note: {e}")

if __name__ == "__main__":
    print("="*60)
    print("AssetOpsBench Data Download")
    print("="*60)
    
    # Download scenarios
    scenarios = download_scenarios()
    
    # Download time series
    download_time_series_data()
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)