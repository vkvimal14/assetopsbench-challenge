"""
LLM Integration Module for AssetOpsBench Challenge
Uses WatsonX AI with LLaMA-3-70B as required by CODS 2025 competition
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

class LLMInterface:
    """Interface for WatsonX AI with LLaMA-3-70B model"""
    
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        
        if not test_mode:
            self.api_key = os.getenv("WATSONX_APIKEY")
            self.project_id = os.getenv("WATSONX_PROJECT_ID") 
            self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            
            if not self.api_key or not self.project_id:
                print("⚠️ WatsonX API credentials not found, using test mode")
                self.test_mode = True
        
        self.model_id = "meta-llama/llama-3-70b-instruct"  # Required by competition
        self.access_token = None
        self.token_expires = None
        
        if not self.test_mode:
            self._get_access_token()
    
    def _get_access_token(self):
        """Get access token for WatsonX API"""
        try:
            token_url = "https://iam.cloud.ibm.com/identity/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "urn:iam:params:oauth:grant-type:apikey",
                "apikey": self.api_key
            }
            
            response = requests.post(token_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Token expires in 1 hour, refresh before that
            self.token_expires = datetime.now().timestamp() + 3300  # 55 minutes
            
        except Exception as e:
            print(f"Error getting access token: {e}")
            raise
    
    def _check_token_validity(self):
        """Check if access token needs refresh"""
        if not self.access_token or datetime.now().timestamp() > self.token_expires:
            self._get_access_token()
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """Generate response using LLaMA-3-70B"""
        if self.test_mode:
            return self._generate_test_response(prompt)
        
        self._check_token_validity()
        
        try:
            generation_url = f"{self.url}/ml/v1/text/generation?version=2023-05-29"
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            
            body = {
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "stop_sequences": ["</response>", "<END>"]
                },
                "model_id": self.model_id,
                "project_id": self.project_id
            }
            
            response = requests.post(generation_url, headers=headers, json=body)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["results"][0]["generated_text"]
            
            # Clean up the response
            generated_text = generated_text.strip()
            if generated_text.endswith("</response>"):
                generated_text = generated_text[:-11].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Error: Could not generate response - {str(e)}"
    
    def _generate_test_response(self, prompt: str) -> str:
        """Generate test response when in test mode"""
        if "iot" in prompt.lower() and "sensor" in prompt.lower():
            return "Based on the IoT system configuration, the available sensors include Supply Temperature, Return Temperature, Condenser Water Flow, and Power sensors for the specified equipment."
        elif "failure" in prompt.lower() or "fsmr" in prompt.lower():
            return "Analysis indicates potential failure modes including overheating, pump malfunction, and electrical faults based on the sensor readings and historical patterns."
        elif "forecast" in prompt.lower() or "tsfm" in prompt.lower():
            return "Time series analysis suggests normal operational patterns with expected values ranging within acceptable parameters over the forecast period."
        elif "work order" in prompt.lower():
            return "Generated work order includes safety procedures, required parts, estimated maintenance duration, and technician assignment based on the identified failure mode."
        else:
            return "LLM analysis completed successfully. The system has processed the request and provided appropriate recommendations based on the available data and context."


class PromptTemplates:
    """Templates for LLM prompts for different agent types"""
    
    @staticmethod
    def iot_agent_prompt(query: str, context: Dict[str, Any]) -> str:
        """Generate prompt for IoT Agent"""
        return f"""
You are an IoT Data Management Agent for industrial asset monitoring systems.

Query: {query}
Context: {json.dumps(context, indent=2)}

Your role is to:
1. Manage sensor data and IoT device configurations
2. Provide access to historical and real-time sensor readings
3. Organize data by sites, assets, and sensor types
4. Ensure data quality and availability

Instructions:
- Use the provided context to understand available sites, assets, and sensors
- Respond with specific, actionable information about IoT system components
- Include relevant details about sensor types, data ranges, and system capabilities
- Format responses to be clear and structured for technical users

Response format: Provide a clear, technical response addressing the query using the context provided.
"""

    @staticmethod 
    def fsmr_agent_prompt(query: str, context: Dict[str, Any]) -> str:
        """Generate prompt for FSMR Agent"""
        return f"""
You are a Failure Modes & Root Cause Analysis Agent for industrial asset maintenance.

Query: {query}
Context: {json.dumps(context, indent=2)}

Your role is to:
1. Identify potential failure modes for equipment and sensors
2. Map sensor readings to possible failure conditions
3. Perform root cause analysis on system anomalies
4. Provide diagnostic insights for maintenance planning

Instructions:
- Analyze the query in the context of failure mode identification
- Consider sensor types, equipment characteristics, and failure patterns
- Provide technical explanations linking symptoms to root causes
- Suggest specific diagnostic approaches and monitoring strategies

Response format: Provide detailed failure analysis with specific failure modes and diagnostic recommendations.
"""

    @staticmethod
    def tsfm_agent_prompt(query: str, context: Dict[str, Any]) -> str:
        """Generate prompt for TSFM Agent"""
        return f"""
You are a Time Series Forecasting & Monitoring Agent for predictive maintenance.

Query: {query}
Context: {json.dumps(context, indent=2)}

Your role is to:
1. Generate forecasts for sensor values and equipment performance
2. Detect anomalies in time series data patterns
3. Identify trends and seasonal patterns in asset operation
4. Provide predictive insights for maintenance scheduling

Instructions:
- Focus on time series analysis techniques and pattern recognition
- Consider seasonal variations, operational cycles, and trend analysis
- Provide specific forecasting methodologies and confidence intervals
- Include anomaly detection thresholds and statistical measures

Response format: Provide forecasting analysis with statistical insights and predictive recommendations.
"""

    @staticmethod
    def wo_agent_prompt(query: str, context: Dict[str, Any]) -> str:
        """Generate prompt for Work Order Agent"""
        return f"""
You are a Work Order Management Agent for industrial maintenance operations.

Query: {query}
Context: {json.dumps(context, indent=2)}

Your role is to:
1. Generate comprehensive work orders for maintenance tasks
2. Determine required parts, tools, and technician skills
3. Estimate maintenance duration and complexity
4. Include safety procedures and compliance requirements

Instructions:
- Create detailed work orders with specific procedures and requirements
- Consider equipment type, failure mode, and maintenance complexity
- Include safety protocols, required certifications, and risk assessments
- Provide realistic time estimates and resource allocation

Response format: Provide structured work order details with safety considerations and resource requirements.
"""

    @staticmethod
    def supervisor_prompt(query: str, available_agents: List[str], context: Dict[str, Any]) -> str:
        """Generate prompt for Supervisor Agent coordination"""
        return f"""
You are a Supervisor Agent coordinating multiple specialized agents for asset management.

Query: {query}
Available Agents: {', '.join(available_agents)}
Context: {json.dumps(context, indent=2)}

Your role is to:
1. Analyze queries to determine which agents should be involved
2. Coordinate multi-agent workflows for complex tasks
3. Integrate responses from multiple agents into comprehensive solutions
4. Ensure efficient agent utilization and avoid redundancy

Instructions:
- Determine the most appropriate agent(s) for the given query
- Design efficient workflows that leverage agent specializations
- Consider dependencies between different agent capabilities
- Provide coordination strategy for optimal task completion

Response format: Provide agent coordination plan with workflow strategy and expected outcomes.
"""