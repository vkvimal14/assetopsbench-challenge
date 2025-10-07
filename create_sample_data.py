"""
Create sample data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_chiller_data():
    """Create sample chiller data"""
    
    # Create 1000 hourly datapoints
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(1000)]
    
    # Generate realistic chiller data
    np.random.seed(42)
    
    # Base pattern: daily cycle + trend + noise
    hours = np.array([d.hour for d in dates])
    days = np.arange(len(dates)) / 24
    
    # Chiller 9 Condenser Water Flow
    base_flow = 500  # GPM
    daily_variation = 50 * np.sin(2 * np.pi * hours / 24)
    trend = 2 * days  # slight upward trend
    noise = np.random.normal(0, 10, len(dates))
    
    water_flow = base_flow + daily_variation + trend + noise
    
    # Other parameters
    inlet_temp = 85 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(dates))
    outlet_temp = inlet_temp - 10 + np.random.normal(0, 1, len(dates))
    pressure = 30 + 2 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.5, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': dates,
        'Chiller 9 Condenser Water Flow': water_flow,
        'Chiller 9 Condenser Inlet Temp': inlet_temp,
        'Chiller 9 Condenser Outlet Temp': outlet_temp,
        'Chiller 9 Condenser Pressure': pressure
    })
    
    # Save to CSV
    df.to_csv('./data/chiller9_annotated_small_test.csv', index=False)
    print(f"✓ Created sample data: {len(df)} rows")
    print(f"✓ Saved to: ./data/chiller9_annotated_small_test.csv")
    
    return df

if __name__ == "__main__":
    print("Creating sample chiller data...")
    df = create_chiller_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData statistics:")
    print(df.describe())