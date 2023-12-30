import unittest
import pandas as pd
import numpy as np
from optimization_model import SupplyChainOptimizer
from data_processor import DataProcessor


class TestSupplyChainOptimizer(unittest.TestCase):
    """Unit tests for SupplyChainOptimizer class"""

  

    def test_capacity_constraints(self):
        """Test that capacity constraints are respected"""
        results = self.optimizer.optimize(self.sample_data)

        # Check that optimization succeeded
        self.assertEqual(results['status'], 'Optimal')

        # Calculate total shipped from each supplier
        allocation_df = pd.DataFrame(results['allocation'])
        supplier_totals = allocation_df.groupby('Supplier')['Quantity'].sum()

        # Check that no supplier exceeds capacity
        for supplier in supplier_totals.index:
            capacity = self.sample_data[
                self.sample_data['Supplier'] == supplier
            ]['Supplier_Capacity'].iloc[0]
            self.assertLessEqual(supplier_totals[supplier], capacity,
                               f"Supplier {supplier} exceeds capacity")

    def test_demand_constraints(self):
        """Test that demand constraints are satisfied"""
        results = self.optimizer.optimize(self.sample_data)

        # Check that optimization succeeded
        self.assertEqual(results['status'], 'Optimal')

        # Calculate total received at each destination
        allocation_df = pd.DataFrame(results['allocation'])
        destination_totals = allocation_df.groupby('Destination')['Quantity'].sum()

        # Check that all demand is met
        for destination in destination_totals.index:
            demand = self.sample_data[
                self.sample_data['Destination'] == destination
            ]['Destination_Demand'].iloc[0]
            self.assertGreaterEqual(destination_totals[destination], demand,
                                  f"Destination {destination} demand not met")

    def test_cost_function_correctness(self):
        """Test that total cost is calculated correctly"""
        results = self.optimizer.optimize(self.sample_data)

        # Check that optimization succeeded
        self.assertEqual(results['status'], 'Optimal')

        # Manually calculate expected cost
        allocation_df = pd.DataFrame(results['allocation'])
        expected_cost = 0

        for _, row in allocation_df.iterrows():
            supplier = row['Supplier']
            destination = row['Destination']
            quantity = row['Quantity']

            # Find cost per unit for this route
            cost_per_unit = self.sample_data[
                (self.sample_data['Supplier'] == supplier) &
                (self.sample_data['Destination'] == destination)
            ]['Cost_per_Unit'].iloc[0]

            expected_cost += quantity * cost_per_unit

        # Check that calculated cost matches
        self.assertAlmostEqual(results['total_cost'], expected_cost, places=2)

    def test_infeasible_problem(self):
        """Test handling of infeasible problems"""
        # Create data where total demand > total capacity
        infeasible_data = self.sample_data.copy()
        infeasible_data['Destination_Demand'] *= 10  # Increase demand significantly

        results = self.optimizer.optimize(infeasible_data)

        # Should not be optimal
        self.assertNotEqual(results['status'], 'Optimal')

    def test_empty_allocation_handling(self):
        """Test handling of cases with no feasible allocation"""
        # Create minimal data that might result in no allocation
        minimal_data = pd.DataFrame({
            'Supplier': ['S1'],
            'Destination': ['D1'],
            'Cost_per_Unit': [10.0],
            'Supplier_Capacity': [0],  # Zero capacity
            'Destination_Demand': [100]
        })

        results = self.optimizer.optimize(minimal_data)

        # Should handle gracefully
        self.assertIsInstance(results, dict)
        self.assertIn('status', results)

    def test_scenario_comparison(self):
        """Test scenario comparison functionality"""
        base_data = self.sample_data.copy()
        scenarios = [
            {'name': 'High Demand', 'demand_multiplier': 1.5},
            {'name': 'Low Cost', 'cost_multiplier': 0.8}
        ]

        results = self.optimizer.compare_scenarios(base_data, scenarios)

        # Check structure
        self.assertIn('baseline', results)
        self.assertIn('scenarios', results)
        self.assertEqual(len(results['scenarios']), 2)

        # Check that scenarios have names
        for scenario in results['scenarios']:
            self.assertIn('scenario_name', scenario)

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality"""
        # First optimize to get a solution
        results = self.optimizer.optimize(self.sample_data)
        self.assertEqual(results['status'], 'Optimal')

        # Then perform sensitivity analysis
        sensitivity = self.optimizer.get_sensitivity_analysis()

        # Check structure
        self.assertIn('capacity_utilization', sensitivity)
        self.assertIn('demand_satisfaction', sensitivity)
        self.assertIn('shadow_prices', sensitivity)

    def test_data_validation_in_optimizer(self):
        """Test that optimizer validates input data"""
        # Test with invalid data (missing columns)
        invalid_data = pd.DataFrame({
            'Supplier': ['S1'],
            'Destination': ['D1']
            # Missing required columns
        })

        # The optimizer should handle invalid data gracefully
        results = self.optimizer.optimize(invalid_data)
        self.assertNotEqual(results['status'], 'Optimal')


if __name__ == '__main__':
    unittest.main()