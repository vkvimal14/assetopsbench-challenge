"""
AssetOpsBench Challenge - Complete Agent System
Save this file as: main_solution.py

Author: Solution for CODS 2025 Agentic AI Challenge
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for the AssetOpsBench system"""
    
    # Watsonx.ai API Configuration
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "your_api_key_here")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your_project_id")
    
    # Model configurations
    DEFAULT_MODEL = "meta-llama/llama-3-70b-instruct"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2000
    
    # Dataset paths
    DATA_DIR = "./data"
    SCENARIOS_PATH = f"{DATA_DIR}/scenarios.csv"
    CHILLER_DATA_PATH = f"{DATA_DIR}/chiller9_annotated_small_test.csv"


# ============================================
# BASE AGENT CLASS
# ============================================

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []
        
    def log(self, message: str):
        """Log agent actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {self.name}: {message}"
        self.memory.append(log_entry)
        print(log_entry)
        
    def clear_memory(self):
        """Clear agent memory"""
        self.memory = []


# ============================================
# SPECIALIZED AGENTS
# ============================================

class IoTAgent(BaseAgent):
    """Agent for handling IoT sensor data and queries"""
    
    def __init__(self):
        super().__init__("IoT Agent", "Sensor Data Management")
        self.sensor_mapping = self._load_sensor_mapping()
        
    def _load_sensor_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """Load mapping of equipment failures to relevant sensors"""
        return {
            "Chiller 6": {
                "Evaporator Water side fouling": [
                    "Chiller 6 Evaporator Water Flow",
                    "Chiller 6 Evaporator Inlet Water Temperature",
                    "Chiller 6 Evaporator Outlet Water Temperature",
                    "Chiller 6 Evaporator Approach Temperature",
                    "Chiller 6 Differential Pressure"
                ],
                "Condenser fouling": [
                    "Chiller 6 Condenser Water Flow",
                    "Chiller 6 Condenser Inlet Water Temperature",
                    "Chiller 6 Condenser Outlet Water Temperature"
                ],
                "Refrigerant leakage": [
                    "Chiller 6 Refrigerant Pressure",
                    "Chiller 6 Compressor Power",
                    "Chiller 6 Cooling Capacity"
                ]
            },
            "Chiller 9": {
                "Evaporator Water side fouling": [
                    "Chiller 9 Evaporator Water Flow",
                    "Chiller 9 Evaporator Inlet Water Temperature",
                    "Chiller 9 Evaporator Outlet Water Temperature"
                ],
                "Condenser fouling": [
                    "Chiller 9 Condenser Water Flow",
                    "Chiller 9 Condenser Inlet Water Temperature",
                    "Chiller 9 Condenser Outlet Water Temperature"
                ]
            }
        }
        
    def identify_relevant_sensor(self, equipment: str, failure_mode: str) -> Optional[str]:
        """Identify the most relevant sensor for a specific failure mode"""
        self.log(f"Identifying sensor for {equipment} - {failure_mode}")
        
        if equipment in self.sensor_mapping:
            if failure_mode in self.sensor_mapping[equipment]:
                sensors = self.sensor_mapping[equipment][failure_mode]
                # Return the primary sensor (first in list)
                primary_sensor = sensors[0]
                self.log(f"Primary sensor identified: {primary_sensor}")
                return primary_sensor
                
        self.log("No matching sensor found")
        return None


class TimeSeriesAgent(BaseAgent):
    """Agent for time series forecasting"""
    
    def __init__(self):
        super().__init__("Time Series Agent", "Forecasting & Analysis")
        
    def load_data(self, filepath: str, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
        """Load time series data"""
        self.log(f"Loading data from {filepath}")
        
        if not os.path.exists(filepath):
            self.log(f"Warning: File not found - {filepath}")
            return pd.DataFrame()
            
        df = pd.read_csv(filepath)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        return df
        
    def forecast(self, data: pd.DataFrame, target_column: str, 
                 timestamp_col: str = 'Timestamp', 
                 forecast_steps: int = 24) -> Dict[str, Any]:
        """
        Forecast time series using multiple methods
        Returns forecasted values with confidence intervals
        """
        self.log(f"Forecasting {target_column}")
        
        if data.empty:
            return {"error": "No data provided"}
        
        # Ensure target column exists
        if target_column not in data.columns:
            self.log(f"Error: Column {target_column} not found")
            return {"error": f"Column {target_column} not found"}
            
        # Extract time series
        ts = data[target_column].values
        timestamps = data[timestamp_col].values
        
        # Method 1: Simple Moving Average
        window = min(7, len(ts) // 4)
        if window > 0:
            sma = np.convolve(ts, np.ones(window)/window, mode='valid')
        else:
            sma = ts
        
        # Method 2: Exponential Smoothing
        alpha = 0.3
        ema = [ts[0]]
        for i in range(1, len(ts)):
            ema.append(alpha * ts[i] + (1 - alpha) * ema[-1])
        
        # Generate forecast
        last_value = ts[-1]
        last_ema = ema[-1]
        trend = (ts[-1] - ts[-min(24, len(ts))]) / min(24, len(ts))
        
        forecasts = []
        for step in range(forecast_steps):
            # Combine methods
            forecast = 0.6 * (last_ema + trend * step) + 0.4 * last_value
            forecasts.append(float(forecast))
            
        # Calculate confidence intervals
        std = np.std(ts[-min(100, len(ts)):])
        lower_bound = [f - 1.96 * std for f in forecasts]
        upper_bound = [f + 1.96 * std for f in forecasts]
        
        self.log(f"Forecast completed: {forecast_steps} steps ahead")
        
        return {
            "forecasts": forecasts,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean_historical": float(np.mean(ts)),
            "std_historical": float(std),
            "last_observed": float(last_value),
            "trend": float(trend)
        }
        
    def detect_anomalies(self, data: pd.DataFrame, target_column: str) -> List[int]:
        """Detect anomalies in time series data"""
        self.log(f"Detecting anomalies in {target_column}")
        
        if target_column not in data.columns:
            return []
            
        ts = data[target_column].values
        
        # Use z-score method
        mean = np.mean(ts)
        std = np.std(ts)
        
        if std == 0:
            return []
            
        z_scores = np.abs((ts - mean) / std)
        
        # Anomalies are points with z-score > 3
        anomaly_indices = np.where(z_scores > 3)[0].tolist()
        
        self.log(f"Found {len(anomaly_indices)} anomalies")
        return anomaly_indices


class DataScienceAgent(BaseAgent):
    """Agent for data analysis and feature engineering"""
    
    def __init__(self):
        super().__init__("Data Science Agent", "Analysis & Feature Engineering")
        
    def analyze_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        self.log("Analyzing dataset")
        
        if data.empty:
            return {"error": "Empty dataset"}
        
        analysis = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": {k: str(v) for k, v in data.dtypes.to_dict().items()},
            "missing_values": data.isnull().sum().to_dict(),
            "summary_stats": {}
        }
        
        # Calculate summary stats for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis["summary_stats"][col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max())
            }
            
        self.log("Analysis complete")
        return analysis
        
    def engineer_features(self, data: pd.DataFrame, 
                         timestamp_col: str = 'Timestamp') -> pd.DataFrame:
        """Create engineered features for better predictions"""
        self.log("Engineering features")
        
        df = data.copy()
        
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Time-based features
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['month'] = df[timestamp_col].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
        # Rolling statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'month', 'is_weekend']:
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7, min_periods=1).std()
                
        self.log(f"Created {len(df.columns) - len(data.columns)} new features")
        return df


class WorkOrderAgent(BaseAgent):
    """Agent for generating and managing work orders"""
    
    def __init__(self):
        super().__init__("Work Order Agent", "Maintenance Planning")
        
    def generate_work_order(self, equipment: str, failure_mode: str, 
                          sensor_data: Optional[Dict] = None,
                          anomalies: Optional[List] = None) -> Dict[str, Any]:
        """Generate a detailed work order"""
        self.log(f"Generating work order for {equipment}")
        
        severity = self._assess_severity(sensor_data, anomalies)
        priority = self._determine_priority(severity)
        
        work_order = {
            "id": f"WO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "equipment": equipment,
            "failure_mode": failure_mode,
            "severity": severity,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "PENDING",
            "recommended_actions": self._get_recommended_actions(failure_mode),
            "estimated_downtime": self._estimate_downtime(failure_mode, severity),
            "required_parts": self._get_required_parts(equipment, failure_mode),
            "sensor_readings": sensor_data,
            "anomaly_count": len(anomalies) if anomalies else 0
        }
        
        self.log(f"Work order {work_order['id']} generated with {priority} priority")
        return work_order
        
    def _assess_severity(self, sensor_data: Optional[Dict], 
                        anomalies: Optional[List]) -> str:
        """Assess failure severity"""
        if anomalies and len(anomalies) > 10:
            return "HIGH"
        elif anomalies and len(anomalies) > 5:
            return "MEDIUM"
        return "LOW"
        
    def _determine_priority(self, severity: str) -> str:
        """Determine work order priority"""
        priority_map = {"HIGH": "URGENT", "MEDIUM": "HIGH", "LOW": "NORMAL"}
        return priority_map.get(severity, "NORMAL")
        
    def _get_recommended_actions(self, failure_mode: str) -> List[str]:
        """Get recommended maintenance actions"""
        actions = {
            "Evaporator Water side fouling": [
                "Inspect evaporator tubes for fouling",
                "Perform chemical cleaning if necessary",
                "Check water quality parameters",
                "Verify water treatment system operation"
            ],
            "Condenser fouling": [
                "Inspect condenser tubes",
                "Clean condenser if required",
                "Check cooling tower water quality"
            ],
            "Refrigerant leakage": [
                "Perform leak detection test",
                "Repair identified leaks",
                "Recharge refrigerant to proper level",
                "Verify system pressures"
            ]
        }
        return actions.get(failure_mode, ["Perform standard inspection"])
        
    def _estimate_downtime(self, failure_mode: str, severity: str) -> str:
        """Estimate maintenance downtime"""
        base_times = {
            "Evaporator Water side fouling": 4,
            "Condenser fouling": 3,
            "Refrigerant leakage": 6
        }
        base = base_times.get(failure_mode, 2)
        multiplier = {"HIGH": 1.5, "MEDIUM": 1.2, "LOW": 1.0}
        hours = base * multiplier[severity]
        return f"{hours:.1f} hours"
        
    def _get_required_parts(self, equipment: str, failure_mode: str) -> List[str]:
        """Get list of potentially required parts"""
        parts = {
            "Evaporator Water side fouling": [
                "Chemical cleaning solution",
                "Gaskets",
                "O-rings"
            ],
            "Refrigerant leakage": [
                "Refrigerant charge",
                "Leak sealant",
                "Pressure gauges"
            ]
        }
        return parts.get(failure_mode, [])


# ============================================
# SUPERVISOR AGENT
# ============================================

class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates all other agents"""
    
    def __init__(self):
        super().__init__("Supervisor Agent", "Coordination & Orchestration")
        self.iot_agent = IoTAgent()
        self.ts_agent = TimeSeriesAgent()
        self.ds_agent = DataScienceAgent()
        self.wo_agent = WorkOrderAgent()
        
    def solve_scenario(self, scenario_id: int, question: str) -> Dict[str, Any]:
        """Main method to solve a scenario"""
        self.log(f"Solving Scenario {scenario_id}")
        self.log(f"Question: {question}")
        
        # Parse the question to understand the task
        task_type = self._identify_task_type(question)
        
        if task_type == "sensor_identification":
            return self._handle_sensor_query(question)
        elif task_type == "forecasting":
            return self._handle_forecasting_query(question)
        elif task_type == "anomaly_detection":
            return self._handle_anomaly_detection(question)
        elif task_type == "root_cause_analysis":
            return self._handle_root_cause_analysis(question)
        else:
            return {"error": "Unknown task type", "question": question}
            
    def _identify_task_type(self, question: str) -> str:
        """Identify the type of task from the question"""
        question_lower = question.lower()
        
        if "sensor" in question_lower and "monitoring" in question_lower:
            return "sensor_identification"
        elif "forecast" in question_lower or "predict" in question_lower:
            return "forecasting"
        elif "anomaly" in question_lower or "detect" in question_lower:
            return "anomaly_detection"
        elif "root cause" in question_lower or "why" in question_lower:
            return "root_cause_analysis"
        else:
            return "unknown"
            
    def _handle_sensor_query(self, question: str) -> Dict[str, Any]:
        """Handle sensor identification queries"""
        self.log("Task: Sensor Identification")
        
        # Parse equipment and failure mode from question
        equipment = self._extract_equipment(question)
        failure_mode = self._extract_failure_mode(question)
        
        # Use IoT Agent to identify sensor
        sensor = self.iot_agent.identify_relevant_sensor(equipment, failure_mode)
        
        return {
            "scenario_type": "sensor_identification",
            "equipment": equipment,
            "failure_mode": failure_mode,
            "recommended_sensor": sensor,
            "reasoning": f"For {failure_mode} in {equipment}, {sensor} provides the most direct measurement of the condition."
        }
        
    def _handle_forecasting_query(self, question: str) -> Dict[str, Any]:
        """Handle forecasting queries"""
        self.log("Task: Time Series Forecasting")
        
        # Extract parameters from question
        target_column = self._extract_target_column(question)
        filepath = self._extract_filepath(question)
        timestamp_col = self._extract_timestamp_col(question)
        
        # Load data using Time Series Agent
        data = self.ts_agent.load_data(filepath, timestamp_col)
        
        if data.empty:
            return {"error": "Could not load data", "filepath": filepath}
        
        # Perform analysis using Data Science Agent
        analysis = self.ds_agent.analyze_dataset(data)
        
        # Engineer features
        data_engineered = self.ds_agent.engineer_features(data, timestamp_col)
        
        # Generate forecast
        forecast_result = self.ts_agent.forecast(data_engineered, target_column, timestamp_col)
        
        # Detect anomalies
        anomalies = self.ts_agent.detect_anomalies(data, target_column)
        
        return {
            "scenario_type": "forecasting",
            "target_column": target_column,
            "data_summary": {
                "rows": len(data),
                "time_range": f"{data[timestamp_col].min()} to {data[timestamp_col].max()}",
                "missing_values": analysis["missing_values"].get(target_column, 0)
            },
            "forecast": forecast_result,
            "anomalies_detected": len(anomalies),
            "anomaly_indices": anomalies[:10]  # First 10 anomalies
        }
        
    def _handle_anomaly_detection(self, question: str) -> Dict[str, Any]:
        """Handle anomaly detection queries"""
        self.log("Task: Anomaly Detection")
        
        target_column = self._extract_target_column(question)
        filepath = self._extract_filepath(question)
        
        data = self.ts_agent.load_data(filepath)
        
        if data.empty:
            return {"error": "Could not load data"}
            
        anomalies = self.ts_agent.detect_anomalies(data, target_column)
        
        # Generate work order if anomalies found
        if len(anomalies) > 5:
            equipment = self._extract_equipment(question)
            work_order = self.wo_agent.generate_work_order(
                equipment, "Anomalous behavior detected", 
                sensor_data={"anomaly_count": len(anomalies)},
                anomalies=anomalies
            )
        else:
            work_order = None
            
        return {
            "scenario_type": "anomaly_detection",
            "anomalies_found": len(anomalies),
            "anomaly_indices": anomalies,
            "work_order": work_order
        }
        
    def _handle_root_cause_analysis(self, question: str) -> Dict[str, Any]:
        """Handle root cause analysis queries"""
        self.log("Task: Root Cause Analysis")
        
        equipment = self._extract_equipment(question)
        
        return {
            "scenario_type": "root_cause_analysis",
            "equipment": equipment,
            "analysis": "Root cause analysis requires domain-specific reasoning",
            "recommended_actions": self.wo_agent._get_recommended_actions("General maintenance")
        }
        
    # Helper methods for parsing questions
    def _extract_equipment(self, question: str) -> str:
        """Extract equipment name from question"""
        import re
        match = re.search(r'Chiller \d+', question)
        return match.group(0) if match else "Unknown Equipment"
        
    def _extract_failure_mode(self, question: str) -> str:
        """Extract failure mode from question"""
        if "fouling" in question.lower():
            if "evaporator" in question.lower():
                return "Evaporator Water side fouling"
            elif "condenser" in question.lower():
                return "Condenser fouling"
        elif "leakage" in question.lower():
            return "Refrigerant leakage"
        return "Unknown failure mode"
        
    def _extract_target_column(self, question: str) -> str:
        """Extract target column from question"""
        import re
        # Look for column name in quotes
        match = re.search(r"'([^']+)'", question)
        if match:
            return match.group(1)
        return "Unknown Column"
        
    def _extract_filepath(self, question: str) -> str:
        """Extract filepath from question"""
        import re
        match = re.search(r"'([^']+\.csv)'", question)
        if match:
            filename = match.group(1)
            return os.path.join(Config.DATA_DIR, filename)
        return ""
        
    def _extract_timestamp_col(self, question: str) -> str:
        """Extract timestamp column name"""
        if "parameter" in question.lower() and "timestamp" in question.lower():
            import re
            match = re.search(r"parameter '([^']+)'", question, re.IGNORECASE)
            if match:
                col_name = match.group(1)
                # Return the original case version
                return col_name if col_name[0].isupper() else col_name.capitalize()
        return "Timestamp"


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("AssetOpsBench Challenge - Agent System")
    print("=" * 60)
    print()
    
    # Initialize supervisor agent
    supervisor = SupervisorAgent()
    
    # Example Scenario 1: Sensor Identification
    print("\n" + "="*60)
    print("SCENARIO 113: Sensor Identification")
    print("="*60)
    
    scenario_1 = {
        "id": 113,
        "question": "If Evaporator Water side fouling occurs for Chiller 6, which sensor is most relevant for monitoring this specific failure?"
    }
    
    result_1 = supervisor.solve_scenario(scenario_1["id"], scenario_1["question"])
    print("\nResult:")
    print(json.dumps(result_1, indent=2))
    
    # Example Scenario 2: Forecasting
    print("\n" + "="*60)
    print("SCENARIO 217: Time Series Forecasting")
    print("="*60)
    
    scenario_2 = {
        "id": 217,
        "question": "Forecast 'Chiller 9 Condenser Water Flow' using data in 'chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp."
    }
    
    # Note: This requires the actual data file to be present
    try:
        result_2 = supervisor.solve_scenario(scenario_2["id"], scenario_2["question"])
        print("\nResult:")
        print(json.dumps(result_2, indent=2, default=str))
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: Ensure data files are present in the ./data directory")
    
    print("\n" + "="*60)
    print("Agent System Execution Complete")
    print("="*60)


if __name__ == "__main__":
    main()