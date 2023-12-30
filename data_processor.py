import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import random


class DataProcessor:

    @staticmethod
    def generate_sample_data(
        num_suppliers: int = 3, num_destinations: int = 4, seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate sample supply chain data for testing and demonstration.

        Args:
            num_suppliers: Number of suppliers to generate
            num_destinations: Number of destinations to generate
            seed: Random seed for reproducible results

        Returns:
            DataFrame with sample supply chain data
        """
        np.random.seed(seed)
        random.seed(seed)

        # Generate supplier and destination names
        suppliers = [f"Supplier_{i+1}" for i in range(num_suppliers)]
        destinations = [f"Destination_{i+1}" for i in range(num_destinations)]

        # Generate capacities and demands
        supplier_capacities = {
            supplier: np.random.randint(80, 200) for supplier in suppliers
        }

        destination_demands = {
            destination: np.random.randint(40, 120) for destination in destinations
        }

        # Ensure total capacity >= total demand
        total_demand = sum(destination_demands.values())
        total_capacity = sum(supplier_capacities.values())

        if total_capacity < total_demand:
            # Adjust capacities proportionally
            multiplier = (total_demand * 1.1) / total_capacity
            supplier_capacities = {
                k: int(v * multiplier) for k, v in supplier_capacities.items()
            }

        # Generate cost matrix (distance-based with some randomness)
        data_rows = []

        for i, supplier in enumerate(suppliers):
            for j, destination in enumerate(destinations):
                # Base cost with distance simulation
                base_cost = 5 + abs(i - j) * 2
                # Add randomness
                cost_variation = np.random.uniform(0.8, 1.3)
                cost_per_unit = round(base_cost * cost_variation, 2)

                data_rows.append(
                    {
                        "Supplier": supplier,
                        "Destination": destination,
                        "Cost_per_Unit": cost_per_unit,
                        "Supplier_Capacity": supplier_capacities[supplier],
                        "Destination_Demand": destination_demands[destination],
                    }
                )

        return pd.DataFrame(data_rows)

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate supply chain data format and constraints.

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of (is_valid, message)
        """
        required_columns = [
            "Supplier",
            "Destination",
            "Cost_per_Unit",
            "Supplier_Capacity",
            "Destination_Demand",
        ]

        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check for empty data
        if data.empty:
            return False, "Data is empty"

        # Check for null values
        if data[required_columns].isnull().any().any():
            return False, "Data contains null values"

        # Check for negative values
        numeric_columns = ["Cost_per_Unit", "Supplier_Capacity", "Destination_Demand"]
        for col in numeric_columns:
            if (data[col] < 0).any():
                return False, f"Negative values found in {col}"

        # Check feasibility (total capacity >= total demand)
        total_capacity = data.groupby("Supplier")["Supplier_Capacity"].first().sum()
        total_demand = data.groupby("Destination")["Destination_Demand"].first().sum()

        if total_capacity < total_demand:
            return (
                False,
                f"Total capacity ({total_capacity}) < Total demand ({total_demand})",
            )

        # Check for duplicate routes
        route_counts = data.groupby(["Supplier", "Destination"]).size()
        if (route_counts > 1).any():
            return False, "Duplicate supplier-destination combinations found"

        return True, "Data validation passed successfully"

    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and clean supply chain data.

        Args:
            data: Raw supply chain data

        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying original data
        processed_data = data.copy()

        # Remove any duplicate rows
        processed_data = processed_data.drop_duplicates()

        # Sort for consistent ordering
        processed_data = processed_data.sort_values(["Supplier", "Destination"])

        # Round numeric values to appropriate precision
        processed_data["Cost_per_Unit"] = processed_data["Cost_per_Unit"].round(2)
        processed_data["Supplier_Capacity"] = processed_data["Supplier_Capacity"].round(
            0
        )
        processed_data["Destination_Demand"] = processed_data[
            "Destination_Demand"
        ].round(0)

        # Reset index
        processed_data = processed_data.reset_index(drop=True)

        return processed_data

    @staticmethod
    def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for supply chain data.

        Args:
            data: Supply chain DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "basic_info": {
                "total_routes": len(data),
                "num_suppliers": data["Supplier"].nunique(),
                "num_destinations": data["Destination"].nunique(),
                "suppliers": sorted(data["Supplier"].unique().tolist()),
                "destinations": sorted(data["Destination"].unique().tolist()),
            },
            "capacity_demand": {
                "total_capacity": data.groupby("Supplier")["Supplier_Capacity"]
                .first()
                .sum(),
                "total_demand": data.groupby("Destination")["Destination_Demand"]
                .first()
                .sum(),
                "capacity_utilization_needed": 0,
                "feasible": True,
            },
            "cost_statistics": {
                "min_cost": data["Cost_per_Unit"].min(),
                "max_cost": data["Cost_per_Unit"].max(),
                "avg_cost": data["Cost_per_Unit"].mean(),
                "median_cost": data["Cost_per_Unit"].median(),
                "std_cost": data["Cost_per_Unit"].std(),
            },
            "supplier_analysis": {},
            "destination_analysis": {},
        }

        # Calculate capacity utilization needed
        total_demand = summary["capacity_demand"]["total_demand"]
        total_capacity = summary["capacity_demand"]["total_capacity"]

        if total_capacity > 0:
            summary["capacity_demand"]["capacity_utilization_needed"] = (
                total_demand / total_capacity
            ) * 100

        summary["capacity_demand"]["feasible"] = total_capacity >= total_demand

        # Supplier analysis
        supplier_data = (
            data.groupby("Supplier")
            .agg(
                {
                    "Supplier_Capacity": "first",
                    "Cost_per_Unit": ["mean", "min", "max"],
                    "Destination": "count",
                }
            )
            .round(2)
        )

        for supplier in summary["basic_info"]["suppliers"]:
            supplier_info = supplier_data.loc[supplier]
            summary["supplier_analysis"][supplier] = {
                "capacity": supplier_info[("Supplier_Capacity", "first")],
                "avg_cost": supplier_info[("Cost_per_Unit", "mean")],
                "min_cost": supplier_info[("Cost_per_Unit", "min")],
                "max_cost": supplier_info[("Cost_per_Unit", "max")],
                "num_destinations": supplier_info[("Destination", "count")],
            }

        # Destination analysis
        destination_data = (
            data.groupby("Destination")
            .agg(
                {
                    "Destination_Demand": "first",
                    "Cost_per_Unit": ["mean", "min", "max"],
                    "Supplier": "count",
                }
            )
            .round(2)
        )

        for destination in summary["basic_info"]["destinations"]:
            dest_info = destination_data.loc[destination]
            summary["destination_analysis"][destination] = {
                "demand": dest_info[("Destination_Demand", "first")],
                "avg_cost": dest_info[("Cost_per_Unit", "mean")],
                "min_cost": dest_info[("Cost_per_Unit", "min")],
                "max_cost": dest_info[("Cost_per_Unit", "max")],
                "num_suppliers": dest_info[("Supplier", "count")],
            }

        return summary

    @staticmethod
    def create_cost_matrix(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a cost matrix from supply chain data.

        Args:
            data: Supply chain DataFrame

        Returns:
            Cost matrix with suppliers as rows and destinations as columns
        """
        cost_matrix = data.pivot_table(
            index="Supplier",
            columns="Destination",
            values="Cost_per_Unit",
            fill_value=np.inf,  # Use infinity for non-existent routes
        )

        return cost_matrix

    @staticmethod
    def export_template() -> pd.DataFrame:
        """
        Generate a template DataFrame for users to fill in their own data.

        Returns:
            Empty template DataFrame with correct structure
        """
        template = pd.DataFrame(
            {
                "Supplier": ["S1", "S1", "S2", "S2"],
                "Destination": ["D1", "D2", "D1", "D2"],
                "Cost_per_Unit": [10.0, 15.0, 12.0, 8.0],
                "Supplier_Capacity": [100, 100, 150, 150],
                "Destination_Demand": [80, 80, 90, 90],
            }
        )

        return template
