# Supply Chain Logistics Optimization Using Linear Programming

## 🚚 Project Overview

This project implements a comprehensive supply chain logistics optimization system using linear programming techniques. The system minimizes transportation and distribution costs while satisfying demand and capacity constraints, providing an interactive dashboard for scenario analysis and decision-making.

## ✨ Features

- **Linear Programming Optimization**: Uses PuLP solver for optimal resource allocation
- **Interactive Dashboard**: Streamlit-based web interface for easy data input and visualization
- **Multi-Scenario Analysis**: Compare different scenarios with adjustable parameters
- **Real-time Visualization**: Interactive charts and heatmaps for result analysis
- **Data Validation**: Comprehensive input validation and error handling
- **Export Functionality**: Download optimization results as CSV files



- Minimize transportation and distribution costs
- Meet all destination demand requirements
- Respect supplier capacity constraints
- Support multiple suppliers, warehouses, and destinations
- Provide scenario-based analysis with adjustable parameters
- Deliver results through an interactive, user-friendly dashboard

## 📝 Task Description

The Supply Chain Logistics Optimization problem involves determining the optimal allocation of goods from multiple suppliers to multiple destinations while minimizing total transportation costs. This is a classic transportation problem in operations research that can be solved using linear programming.

### Problem Formulation
- **Suppliers**: Have limited capacity and supply goods at different costs
- **Destinations**: Have specific demand requirements that must be met
- **Routes**: Each supplier-destination pair has an associated transportation cost per unit
- **Objective**: Minimize total transportation cost while satisfying all constraints

### Mathematical Model
```
Minimize: ΣᵢΣⱼ Cᵢⱼ × Xᵢⱼ

Subject to:
Σⱼ Xᵢⱼ ≤ Capacityᵢ    ∀ suppliers i  (supply constraints)
Σᵢ Xᵢⱼ ≥ Demandⱼ     ∀ destinations j  (demand constraints)
Xᵢⱼ ≥ 0               ∀ routes (i,j)  (non-negativity)
```

Where:
- Cᵢⱼ = cost per unit from supplier i to destination j
- Xᵢⱼ = quantity shipped from supplier i to destination j
- Capacityᵢ = maximum supply from supplier i
- Demandⱼ = required quantity at destination j

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Optimization**: PuLP (Linear Programming)
- **Data Processing**: pandas, NumPy
- **Web Interface**: Streamlit
- **Visualization**: Matplotlib, Plotly
- **Development Environment**: VS Code

## 📋 System Requirements

- Python 3.8 or higher
- 4GB RAM (minimum)
- Modern web browser

## 🚀 Installation & Setup

### 1. Clone or Download the Project
```bash
cd /home/alerman/projects/suppl_chain_optimization
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Access the Dashboard
Open your web browser and navigate to `http://localhost:8501`

## 📊 Data Format

Your CSV file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Supplier | Supplier identifier | S1, S2, S3 |
| Destination | Destination identifier | D1, D2, D3 |
| Cost_per_Unit | Transportation cost per unit | 10.50 |
| Supplier_Capacity | Maximum capacity of supplier | 100 |
| Destination_Demand | Demand at destination | 75 |

### Sample Data Structure
```csv
Supplier,Destination,Cost_per_Unit,Supplier_Capacity,Destination_Demand
Supplier_1,Destination_1,5.84,179,86
Supplier_1,Destination_2,7.77,179,118
Supplier_2,Destination_1,7.13,163,86
Supplier_2,Destination_2,5.89,163,118
```

## 🎮 How to Use

### Step 1: Data Input
- **Option A**: Use the generated sample data for testing
- **Option B**: Upload your own CSV file with supply chain data
- The system validates data format and constraints automatically

### Step 2: Configure Optimization
- Navigate to the "Optimization" page
- Adjust scenario parameters:
  - **Demand Multiplier**: Scale demand up/down for what-if analysis
  - **Cost Multiplier**: Adjust transportation costs
- Click "Run Optimization" to solve

### Step 3: Analyze Results
- View optimization status and total cost
- Examine the allocation heatmap
- Analyze cost breakdown by supplier
- Review detailed allocation table
- Download results as CSV

### Step 4: Scenario Testing
- Modify parameters and re-run optimization
- Compare different scenarios
- Make data-driven decisions

## � Dashboard Screenshot

![Supply Chain Optimization Dashboard](dashboard_screenshot.png)

*The interactive dashboard showing optimization results with allocation heatmap, cost breakdown charts, and scenario analysis controls.*

## �📈 System Workflow

1. **Data Upload**: User uploads supply chain data (cost matrix, demand, supply, capacity)
2. **Data Validation**: System validates data format and feasibility
3. **Model Formulation**: Creates linear programming problem with constraints
4. **Optimization**: Solver calculates optimal shipping plan
5. **Results Display**: Shows cost breakdown and shipment allocations
6. **Interactive Analysis**: User can adjust parameters and re-run optimization

## 🧮 Optimization Model

### Objective Function
Minimize total transportation cost:
```
Minimize: Σ(Cost_ij × Quantity_ij)
```

### Constraints

1. **Supplier Capacity Constraints**:
   ```
   Σ(Quantity_ij) ≤ Capacity_i  ∀ suppliers i
   ```

2. **Destination Demand Constraints**:
   ```
   Σ(Quantity_ij) ≥ Demand_j  ∀ destinations j
   ```

3. **Non-negativity Constraints**:
   ```
   Quantity_ij ≥ 0  ∀ routes (i,j)
   ```

## 📊 Results & Performance

- **Efficiency**: Solves small-to-medium problems (up to 1000 routes) in seconds
- **Optimization**: Produces optimal shipping allocations with significant cost savings
- **Scalability**: Handles hundreds of routes efficiently on standard hardware
- **Interactivity**: Real-time scenario testing enables quick decision-making

## 🔧 Project Structure

```
suppl_chain_optimization/
├── app.py                 # Main Streamlit application
├── optimization_model.py  # Linear programming optimization logic
├── data_processor.py      # Data validation and preprocessing
├── test_optimization.py   # Unit tests for optimization model
├── sample_data.csv        # Sample supply chain data
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── venv/                 # Virtual environment (if used)
```

## 💡 Example Use Cases

1. **Manufacturing Distribution**: Optimize product distribution from factories to retail stores
2. **Warehouse Management**: Allocate inventory across multiple warehouses to minimize shipping costs
3. **Supply Network Design**: Determine optimal supplier selection and allocation
4. **Logistics Planning**: Plan transportation routes for minimum cost delivery

## 🚀 Future Improvements

- **Multi-objective Optimization**: Add environmental impact (CO₂ emissions) as secondary objective
- **Real-world Integration**: Connect with logistics APIs for live traffic and fuel cost data
- **Stochastic Modeling**: Handle uncertain demand with probability distributions
- **Geographic Visualization**: Add map-based route visualization
- **Advanced Constraints**: Include time windows, vehicle capacity, and routing constraints
- **Machine Learning**: Predict demand patterns and optimize proactively

## 🧪 Testing

The project includes comprehensive unit tests for the optimization model:

```bash
# Run all tests
python -m pytest test_optimization.py -v

# Run with coverage
python -m pytest test_optimization.py --cov=optimization_model --cov-report=html
```

### Test Coverage
- **Capacity Constraints**: Ensures suppliers don't exceed capacity limits
- **Demand Constraints**: Verifies all destination demands are met
- **Cost Function**: Validates correct cost calculation
- **Scenario Analysis**: Tests multi-scenario comparisons
- **Edge Cases**: Handles infeasible problems and invalid data

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Format Errors**: Check that your CSV matches the required format exactly

3. **Infeasible Solutions**: Verify that total supplier capacity ≥ total demand

4. **Performance Issues**: For large datasets, consider breaking into smaller problems

### Error Messages

- **"Total capacity < Total demand"**: Increase supplier capacities or reduce demands
- **"Data validation failed"**: Check for missing columns or negative values
- **"Optimization failed"**: Review data constraints and feasibility

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the data format requirements
3. Ensure all dependencies are properly installed
4. Test with the provided sample data first

## 📜 License

This project is provided as-is for educational and commercial use. Feel free to modify and extend according to your needs.

## 🎯 Conclusion

This project successfully demonstrates the power of linear programming in optimizing supply chain logistics. The Streamlit-based interface makes complex optimization accessible to users without programming experience, while the modular design allows for easy extension and customization. Whether you're optimizing a small distribution network or planning enterprise-scale logistics, this tool provides a solid foundation for data-driven decision-making.
