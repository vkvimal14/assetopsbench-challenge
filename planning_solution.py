"""
AssetOpsBench Challenge - LLM-Enhanced Agent System (Task Planning Submission)
CODS 2025 Competition Submission with LLaMA-3-70B Integration

This solution is modified to output the SUPERVISOR'S PLAN for the Task Planning track.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from difflib import get_close_matches
import re

# Import our LLM integration
from llm_integration import LLMInterface, PromptTemplates

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for the LLM-enhanced AssetOpsBench system"""
    
    # Model configuration (CODS 2025 requirement)
    REQUIRED_MODEL = "meta-llama/llama-3-70b-instruct"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2000
    
    # Dataset paths
    DATA_DIR = "./data"
    SCENARIOS_PATH = f"{DATA_DIR}/scenarios.csv"
    CHILLER_DATA_PATH = f"{DATA_DIR}/chiller9_annotated_small_test.csv"
    
    # Agent configuration
    USE_LLM_REASONING = True  # Competition requirement
    FALLBACK_TO_RULES = True  # Backup for reliability


# ============================================
# LLM-ENHANCED BASE AGENT
# ============================================

class LLMEnhancedAgent:
    """Base class for LLM-enhanced agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []
        # Try to initialize LLM, fallback to test mode if credentials missing
        try:
            self.llm = LLMInterface()
        except:
            print(f"⚠️ Using test mode for {name}")
            self.llm = LLMInterface(test_mode=True)
        self.knowledge_base = self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize agent's knowledge base"""
        return {
            "sites": ["MAIN"],
            "equipment_types": [
                "Chiller", "Pump", "Motor", "Compressor", "Heat Exchanger",
                "AHU", "VAV", "Boiler", "Cooling Tower", "Fan"
            ],
            "sensor_types": [
                "Temperature", "Pressure", "Flow", "Vibration", "Power",
                "Current", "Voltage", "Humidity", "CO2"
            ],
            "failure_modes": [
                "Bearing failure", "Overheating", "Vibration", "Electrical fault",
                "Sensor malfunction", "Refrigerant leak", "Blockage", "Corrosion"
            ]
        }
    
    def log(self, message: str):
        """Log agent actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {self.name}: {message}"
        self.memory.append(log_entry)
        print(log_entry)
    
    def llm_reasoning(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Use LLM for reasoning with fallback"""
        if not Config.USE_LLM_REASONING:
            return self.fallback_reasoning(prompt, context)
        
        try:
            # Add agent context to the prompt
            enhanced_context = {**(context or {}), **self.knowledge_base}
            response = self.llm.generate_response(prompt)
            self.log(f"LLM reasoning applied for: {prompt[:50]}...")
            return response
        except Exception as e:
            self.log(f"LLM error, using fallback: {e}")
            if Config.FALLBACK_TO_RULES:
                return self.fallback_reasoning(prompt, context)
            else:
                raise
    
    def fallback_reasoning(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Fallback rule-based reasoning"""
        return f"Fallback response for: {prompt}"


# ============================================
# SPECIALIZED LLM-ENHANCED AGENTS (Dummy for Planning)
# ============================================
# For the planning submission, the actual execution logic of these agents is not needed.
# We only need the supervisor's plan.

class LLMIoTAgent(LLMEnhancedAgent):
    def __init__(self):
        super().__init__("IoT Agent", "Sensor Data Management")

class LLMFSMRAgent(LLMEnhancedAgent):
    def __init__(self):
        super().__init__("FSMR Agent", "Failure Modes & Root Cause Analysis")

class LLMTSFMAgent(LLMEnhancedAgent):
    def __init__(self):
        super().__init__("TSFM Agent", "Time Series Forecasting & Monitoring")

class LLMWorkOrderAgent(LLMEnhancedAgent):
    def __init__(self):
        super().__init__("WO Agent", "Work Order Management")


# ============================================
# LLM-ENHANCED SUPERVISOR AGENT (PLANNING MODE)
# ============================================

class LLMSupervisorAgent(LLMEnhancedAgent):
    """LLM-enhanced Supervisor Agent for coordinating multiple agents"""
    
    def __init__(self):
        super().__init__("Supervisor Agent", "Multi-Agent Coordination")
        self.agents = {
            "iot": LLMIoTAgent(),
            "fsmr": LLMFSMRAgent(),
            "tsfm": LLMTSFMAgent(),
            "wo": LLMWorkOrderAgent()
        }
    
    def process_query(self, query: str, scenario_id: int = None) -> Dict[str, Any]:
        """
        MODIFIED FOR TASK PLANNING: This method now returns the coordination plan
        generated by the supervisor's LLM reasoning.
        """
        
        # Use LLM to determine which agents to use and how to coordinate them
        coordination_prompt = PromptTemplates.supervisor_prompt(
            query,
            list(self.agents.keys()),
            {"scenario_id": scenario_id}
        )
        
        coordination_plan = self.llm_reasoning(coordination_prompt)
        
        self.log(f"Processing query: {query[:100]}...")
        self.log(f"LLM coordination plan: {coordination_plan[:200]}...")
        
        # For the planning track, we return the plan itself.
        return {
            "scenario_id": scenario_id,
            "query": query,
            "task_plan": coordination_plan
        }

# ============================================
# MAIN SOLUTION PROCESSOR
# ============================================

class LLMAssetOpsBenchProcessor:
    """Main processor for LLM-enhanced AssetOpsBench solution"""
    
    def __init__(self):
        self.supervisor = LLMSupervisorAgent()
    
    def process_scenarios(self, scenarios_file: str) -> Dict[int, Any]:
        """Process all scenarios from CSV file"""
        try:
            df = pd.read_csv(scenarios_file)
            results = {}
            
            for _, row in df.iterrows():
                scenario_id = int(row['id'])
                query = row['text']
                
                print(f"\n{'='*50}")
                print(f"Generating plan for Scenario {scenario_id}")
                print(f"Query: {query}")
                print(f"{'='*50}")
                
                try:
                    result = self.supervisor.process_query(query, scenario_id)
                    results[scenario_id] = result
                    print(f"✅ Plan for scenario {scenario_id} generated successfully")
                    
                except Exception as e:
                    print(f"❌ Error in scenario {scenario_id}: {e}")
                    results[scenario_id] = {"error": str(e), "query": query}
            
            return results
            
        except Exception as e:
            print(f"Error processing scenarios: {e}")
            return {}
    
    def save_results(self, results: Dict[int, Any], output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    print("🚀 Starting LLM-Enhanced AssetOpsBench Solution (Task Planning Mode)")
    print(f"📋 Using model: {Config.REQUIRED_MODEL}")
    
    # Initialize processor
    processor = LLMAssetOpsBenchProcessor()
    
    # Process scenarios
    scenarios_file = Config.SCENARIOS_PATH
    if os.path.exists(scenarios_file):
        results = processor.process_scenarios(scenarios_file)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"submissions/submission_planning_{timestamp}.json"
        processor.save_results(results, output_file)
        
        print(f"\n📊 EXECUTION SUMMARY")
        print(f"Total plans generated: {len(results)}")
        print(f"Output file: {output_file}")
        
    else:
        print(f"❌ Scenarios file not found: {scenarios_file}")


if __name__ == "__main__":
    main()
