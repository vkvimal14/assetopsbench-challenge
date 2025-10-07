# 🏭 AssetOpsBench Challenge - AI-Powered Industrial Asset Management

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

> **CODS 2025 Agentic AI Challenge Submission**  
> An intelligent multi-agent system for industrial asset operations, predictive maintenance, and anomaly detection.

## 🎯 Project Overview

AssetOpsBench is an advanced AI-powered system designed to manage industrial equipment (chillers, turbines, etc.) through intelligent agents that handle:

- **IoT Sensor Monitoring** 📊
- **Time Series Forecasting** 📈  
- **Anomaly Detection** 🚨
- **Predictive Maintenance** 🔧
- **Work Order Management** 📋

## 🏗️ System Architecture

### Multi-Agent Framework

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Agent     │    │ TimeSeries      │    │ DataScience     │
│                 │    │ Agent           │    │ Agent           │
│ • Sensor Data   │    │ • Forecasting   │    │ • Feature Eng   │
│ • Equipment     │    │ • Anomaly Det   │    │ • Analysis      │
│ • Mappings      │    │ • Trends        │    │ • ML Models     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Supervisor      │
                    │ Agent           │
                    │                 │
                    │ • Coordination  │
                    │ • Task Routing  │
                    │ • Decision      │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ WorkOrder       │    │ Enhanced        │    │ Results &       │
│ Agent           │    │ Agents          │    │ Submission      │
│                 │    │                 │    │                 │
│ • Maintenance   │    │ • Advanced ML   │    │ • JSON Output   │
│ • Scheduling    │    │ • Real-time     │    │ • Performance   │
│ • Optimization  │    │ • Integration   │    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd assetopsbench-challenge
   ```

2. **Create virtual environment**
   ```bash
   python -m venv watson311
   watson311\Scripts\activate  # Windows
   # source watson311/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Usage

#### Run Single Solution
```bash
python main_solution.py
```

#### Run Enhanced Solution
```bash
python enhanced_solution.py
```

#### Process All Scenarios
```bash
python process_all_scenarios.py
```

#### Create Test Data
```bash
python create_sample_data.py
```

## 📊 Features

### Current Capabilities

- ✅ **Sensor Identification**: Maps equipment failures to relevant sensors
- ✅ **Time Series Forecasting**: Predicts equipment performance 
- ✅ **Anomaly Detection**: Identifies unusual patterns in sensor data
- ✅ **Work Order Generation**: Creates maintenance recommendations
- ✅ **Multi-Agent Coordination**: Orchestrates specialized AI agents

### Supported Scenarios (141 total)

| Category | Count | Examples |
|----------|-------|----------|
| IoT Queries | 48 | Site listing, asset metadata, sensor data |
| Knowledge Queries | 41 | Failure modes, sensor mappings |
| Time Series | 23 | Forecasting, anomaly detection |
| Work Orders | 29 | Maintenance planning, optimization |

## 📈 Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Success Rate | 7.8% | 85%+ |
| Response Time | Variable | <2s |
| Accuracy | Basic | 90%+ |
| Scenarios Handled | 11/141 | 120+/141 |

## 🗂️ Project Structure

```
assetopsbench-challenge/
├── 📄 main_solution.py          # Core multi-agent system
├── 📄 enhanced_solution.py      # Improved version with better classification
├── 📄 process_all_scenarios.py  # Batch processing for all scenarios
├── 📁 data/                     # Dataset files
│   ├── scenarios.csv            # All 141 test scenarios
│   └── chiller9_annotated_small_test.csv
├── 📁 submissions/              # Generated submission files
│   ├── submission.json          # Final submission for competition
│   ├── detailed_results.json    # Detailed analysis results
│   └── summary_report.txt       # Performance summary
├── 📁 logs/                     # System logs
├── 📁 results/                  # Processing results
├── 📁 models/                   # ML model artifacts
├── 📁 watson311/                # Virtual environment
├── 📄 requirements.txt          # Python dependencies
└── 📄 README.md                # This file
```

## 🔧 Equipment & Sensors

### Supported Equipment
- **Chiller 6** (CWC04006) - 13 sensors, 7 failure modes
- **Chiller 9** (CWC04009) - 4 sensors, 2 failure modes

### Sensor Types
- Temperature (Supply, Return, Evaporator, Condenser)
- Flow (Water flow rates)
- Pressure (Refrigerant pressure)
- Power (Energy consumption)
- Efficiency metrics

### Failure Modes
- Evaporator Water side fouling
- Condenser fouling  
- Refrigerant leakage
- Compressor overheating
- Motor failures

## 🛠️ Development

### Running Tests
```bash
python test_setup.py
python test_watsonx.py
```

### Adding New Agents
```python
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Custom Agent", "Specialized Task")
    
    def process(self, data):
        # Your agent logic here
        return result
```

### Processing New Scenarios
```python
supervisor = SupervisorAgent()
result = supervisor.solve_scenario(scenario_id, question)
```

## 📋 TODO & Improvements

### 🚨 Critical (Week 1)
- [ ] Fix task classification for all 141 scenarios
- [ ] Implement proper data integration
- [ ] Add error handling and retry logic

### ⚡ High Priority (Week 2)
- [ ] Advanced ML models for forecasting
- [ ] Real-time data integration
- [ ] Enhanced anomaly detection algorithms

### 📈 Medium Priority (Week 3-4)
- [ ] Web dashboard for monitoring
- [ ] API endpoints for external integration
- [ ] Advanced work order optimization

### 🎨 Nice-to-Have (Month 2)
- [ ] Real-time streaming data processing
- [ ] Machine learning model training pipeline
- [ ] Multi-tenant architecture
- [ ] Mobile app interface

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Competition Info

- **Event**: CODS 2025 Agentic AI Challenge
- **Category**: AssetOpsBench
- **Submission Format**: JSON with scenario solutions
- **Evaluation**: Accuracy, completeness, performance

## 📞 Contact & Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: [Your email for support]

---

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ for industrial AI and predictive maintenance*