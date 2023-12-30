import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class SupplyChainOptimizer:
    """
    Linear programming optimizer for supply chain logistics.
    Minimizes transportation costs while satisfying supply and demand constraints.
    """

    def __init__(self):
        self.model = None
        self.variables = {}
        self.suppliers = []
        self.destinations = []
        self.costs = {}
        self.capacities = {}
        self.demands = {}

    def optimize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main optimization function that formulates and solves the linear programming problem.

        Args:
            data (pd.DataFrame): Supply chain data with columns:
                - Supplier: Supplier identifier
                - Destination: Destination identifier
                - Cost_per_Unit: Transportation cost per unit
                - Supplier_Capacity: Maximum capacity of supplier
                - Destination_Demand: Demand at destination

        Returns:
            Dict containing optimization results
        """
        try:
            # Prepare data
            self._prepare_data(data)

            # Create optimization model
            self._create_model()

            # Add constraints
            self._add_constraints()

            # Solve the problem
            status = self.model.solve(pulp.PULP_CBC_CMD(msg=0))

            # Extract and return results
            return self._extract_results(status)

        except Exception as e:
            return {
                "status": "Error",
                "message": str(e),
                "total_cost": 0,
                "allocation": [],
            }

    def _prepare_data(self, data: pd.DataFrame):
        """Prepare and validate input data"""
        # Extract unique suppliers and destinations
        self.suppliers = sorted(data["Supplier"].unique())
        self.destinations = sorted(data["Destination"].unique())

        # Create cost matrix
        self.costs = {}
        for _, row in data.iterrows():
            self.costs[(row["Supplier"], row["Destination"])] = row["Cost_per_Unit"]

        # Extract capacities and demands
        self.capacities = (
            data.groupby("Supplier")["Supplier_Capacity"].first().to_dict()
        )
        self.demands = (
            data.groupby("Destination")["Destination_Demand"].first().to_dict()
        )

        # Validation
        total_capacity = sum(self.capacities.values())
        total_demand = sum(self.demands.values())

        if total_capacity < total_demand:
            raise ValueError(
                f"Total capacity ({total_capacity}) is less than total demand ({total_demand})"
            )

    def _create_model(self):
        """Create the linear programming model with objective function"""
        # Initialize the model
        self.model = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)

        # Create decision variables
        self.variables = {}
        for supplier in self.suppliers:
            for destination in self.destinations:
                if (supplier, destination) in self.costs:
                    var_name = f"ship_{supplier}_to_{destination}"
                    self.variables[(supplier, destination)] = pulp.LpVariable(
                        var_name, lowBound=0, cat="Continuous"
                    )

        # Define objective function: minimize total transportation cost
        objective = pulp.lpSum(
            [
                self.costs[(supplier, destination)]
                * self.variables[(supplier, destination)]
                for supplier in self.suppliers
                for destination in self.destinations
                if (supplier, destination) in self.variables
            ]
        )

        self.model += objective

    def _add_constraints(self):
        """Add all constraints to the model"""
        # Supplier capacity constraints
        for supplier in self.suppliers:
            constraint = (
                pulp.lpSum(
                    [
                        self.variables[(supplier, destination)]
                        for destination in self.destinations
                        if (supplier, destination) in self.variables
                    ]
                )
                <= self.capacities[supplier]
            )

            self.model += constraint, f"Capacity_constraint_{supplier}"

        # Destination demand constraints
        for destination in self.destinations:
            constraint = (
                pulp.lpSum(
                    [
                        self.variables[(supplier, destination)]
                        for supplier in self.suppliers
                        if (supplier, destination) in self.variables
                    ]
                )
                >= self.demands[destination]
            )

            self.model += constraint, f"Demand_constraint_{destination}"

    def _extract_results(self, status: int) -> Dict[str, Any]:
        """Extract and format optimization results"""
        status_map = {
            pulp.LpStatusOptimal: "Optimal",
            pulp.LpStatusInfeasible: "Infeasible",
            pulp.LpStatusUnbounded: "Unbounded",
            pulp.LpStatusNotSolved: "Not Solved",
            pulp.LpStatusUndefined: "Undefined",
        }

        result = {
            "status": status_map.get(status, "Unknown"),
            "total_cost": 0,
            "allocation": [],
            "solver_details": {},
        }

        if status == pulp.LpStatusOptimal:
            # Calculate total cost
            result["total_cost"] = pulp.value(self.model.objective)

            # Extract allocation details
            allocation = []
            for (supplier, destination), variable in self.variables.items():
                quantity = pulp.value(variable)
                if quantity > 0.001:  # Only include non-zero allocations
                    cost_per_unit = self.costs[(supplier, destination)]
                    total_cost = quantity * cost_per_unit

                    allocation.append(
                        {
                            "Supplier": supplier,
                            "Destination": destination,
                            "Quantity": round(quantity, 2),
                            "Cost_per_Unit": cost_per_unit,
                            "Total_Cost": round(total_cost, 2),
                        }
                    )

            result["allocation"] = allocation

            # Add solver details
            result["solver_details"] = {
                "variables_count": len(self.variables),
                "constraints_count": len(self.model.constraints),
                "suppliers_count": len(self.suppliers),
                "destinations_count": len(self.destinations),
            }

        return result

    def get_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on the optimal solution.
        Returns shadow prices and ranges for constraints.
        """
        if not self.model or self.model.status != pulp.LpStatusOptimal:
            return {"error": "No optimal solution available for sensitivity analysis"}

        sensitivity = {
            "shadow_prices": {},
            "capacity_utilization": {},
            "demand_satisfaction": {},
        }

        # Calculate capacity utilization
        for supplier in self.suppliers:
            used_capacity = sum(
                [
                    pulp.value(self.variables[(supplier, destination)])
                    for destination in self.destinations
                    if (supplier, destination) in self.variables
                ]
            )

            sensitivity["capacity_utilization"][supplier] = {
                "used": round(used_capacity, 2),
                "total": self.capacities[supplier],
                "utilization_rate": round(
                    used_capacity / self.capacities[supplier] * 100, 2
                ),
            }

        # Calculate demand satisfaction
        for destination in self.destinations:
            received = sum(
                [
                    pulp.value(self.variables[(supplier, destination)])
                    for supplier in self.suppliers
                    if (supplier, destination) in self.variables
                ]
            )

            sensitivity["demand_satisfaction"][destination] = {
                "received": round(received, 2),
                "demanded": self.demands[destination],
                "satisfaction_rate": round(
                    received / self.demands[destination] * 100, 2
                ),
            }

        return sensitivity

    def compare_scenarios(
        self, base_data: pd.DataFrame, scenarios: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compare multiple scenarios against a baseline.

        Args:
            base_data: Baseline supply chain data
            scenarios: List of scenario modifications

        Returns:
            Comparison results
        """
        results = {"baseline": self.optimize(base_data), "scenarios": []}

        for i, scenario in enumerate(scenarios):
            # Apply scenario modifications
            modified_data = base_data.copy()

            if "demand_multiplier" in scenario:
                modified_data["Destination_Demand"] *= scenario["demand_multiplier"]

            if "cost_multiplier" in scenario:
                modified_data["Cost_per_Unit"] *= scenario["cost_multiplier"]

            if "capacity_multiplier" in scenario:
                modified_data["Supplier_Capacity"] *= scenario["capacity_multiplier"]

            # Optimize scenario
            scenario_result = self.optimize(modified_data)
            scenario_result["scenario_name"] = scenario.get("name", f"Scenario {i+1}")
            scenario_result["modifications"] = scenario

            results["scenarios"].append(scenario_result)

        return results

    def run_predefined_scenarios(self, base_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run predefined common scenarios for supply chain analysis.

        Args:
            base_data: Baseline supply chain data

        Returns:
            Results for all predefined scenarios
        """
        scenarios = [
            {
                "name": "High Demand (+50%)",
                "demand_multiplier": 1.5,
                "description": "Increased demand scenario"
            },
            {
                "name": "Low Demand (-30%)",
                "demand_multiplier": 0.7,
                "description": "Decreased demand scenario"
            },
            {
                "name": "Cost Increase (+25%)",
                "cost_multiplier": 1.25,
                "description": "Higher transportation costs"
            },
            {
                "name": "Cost Reduction (-20%)",
                "cost_multiplier": 0.8,
                "description": "Lower transportation costs"
            },
            {
                "name": "Capacity Expansion (+40%)",
                "capacity_multiplier": 1.4,
                "description": "Increased supplier capacities"
            },
            {
                "name": "Capacity Reduction (-25%)",
                "capacity_multiplier": 0.75,
                "description": "Reduced supplier capacities"
            },
            {
                "name": "Peak Season",
                "demand_multiplier": 1.8,
                "cost_multiplier": 1.3,
                "description": "High demand with higher costs"
            },
            {
                "name": "Economic Downturn",
                "demand_multiplier": 0.6,
                "cost_multiplier": 0.9,
                "description": "Low demand with lower costs"
            }
        ]

        return self.compare_scenarios(base_data, scenarios)

    def analyze_supply_chain_robustness(self, base_data: pd.DataFrame,
                                      disruption_scenarios: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze supply chain robustness under various disruption scenarios.

        Args:
            base_data: Baseline supply chain data
            disruption_scenarios: Custom disruption scenarios (optional)

        Returns:
            Robustness analysis results
        """
        if disruption_scenarios is None:
            disruption_scenarios = [
                {
                    "name": "Supplier Disruption (S1 -50% capacity)",
                    "supplier_disruptions": {"Supplier_1": 0.5},
                    "description": "Major supplier capacity reduction"
                },
                {
                    "name": "Route Disruption (S1->D1 blocked)",
                    "blocked_routes": [("Supplier_1", "Destination_1")],
                    "description": "Specific route becomes unavailable"
                },
                {
                    "name": "Demand Spike (D1 +100%)",
                    "demand_changes": {"Destination_1": 2.0},
                    "description": "Sudden demand increase at one destination"
                },
                {
                    "name": "Cost Surge (+50% all routes)",
                    "cost_multiplier": 1.5,
                    "description": "Across-the-board cost increase"
                }
            ]

        results = {"baseline": self.optimize(base_data), "disruptions": []}

        for scenario in disruption_scenarios:
            modified_data = base_data.copy()

            # Apply supplier disruptions
            if "supplier_disruptions" in scenario:
                for supplier, factor in scenario["supplier_disruptions"].items():
                    if supplier in modified_data["Supplier"].values:
                        modified_data.loc[
                            modified_data["Supplier"] == supplier, "Supplier_Capacity"
                        ] *= factor

            # Apply route blockages (set cost to infinity)
            if "blocked_routes" in scenario:
                for supplier, destination in scenario["blocked_routes"]:
                    mask = (modified_data["Supplier"] == supplier) & \
                           (modified_data["Destination"] == destination)
                    modified_data.loc[mask, "Cost_per_Unit"] = np.inf

            # Apply demand changes
            if "demand_changes" in scenario:
                for destination, factor in scenario["demand_changes"].items():
                    if destination in modified_data["Destination"].values:
                        modified_data.loc[
                            modified_data["Destination"] == destination, "Destination_Demand"
                        ] *= factor

            # Apply cost multiplier
            if "cost_multiplier" in scenario:
                modified_data["Cost_per_Unit"] *= scenario["cost_multiplier"]

            # Run optimization
            disruption_result = self.optimize(modified_data)
            disruption_result.update({
                "scenario_name": scenario["name"],
                "description": scenario["description"],
                "modifications": scenario
            })

            results["disruptions"].append(disruption_result)

        # Calculate robustness metrics
        baseline_cost = results["baseline"].get("total_cost", 0)
        results["robustness_metrics"] = {}

        for disruption in results["disruptions"]:
            disruption_cost = disruption.get("total_cost", 0)
            if baseline_cost > 0:
                cost_increase_pct = ((disruption_cost - baseline_cost) / baseline_cost) * 100
            else:
                cost_increase_pct = 0

            results["robustness_metrics"][disruption["scenario_name"]] = {
                "cost_increase_percentage": round(cost_increase_pct, 2),
                "feasible": disruption.get("status") == "Optimal"
            }

        return results
