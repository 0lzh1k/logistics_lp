"""
Supply Chain Logistics Optimization Using Linear Programming
Interactive Streamlit Dashboard for Supply Chain Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from optimization_model import SupplyChainOptimizer
from data_processor import DataProcessor


def main():
    st.set_page_config(
        page_title="Supply Chain Optimization Dashboard", page_icon=None, layout="wide"
    )

    st.title("Supply Chain Logistics Optimization")
    st.markdown("### Using Linear Programming to Minimize Transportation Costs")

    # Sidebar for navigation and parameters
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", ["Home", "Data Upload", "Optimization", "Results Analysis"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Optimization":
        show_optimization_page()
    elif page == "Results Analysis":
        show_results_page()


def show_home_page():
    st.header("Welcome to Supply Chain Optimization Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Overview")
        st.markdown(
            """
        This application helps optimize supply chain logistics by:
        - **Minimizing transportation costs**
        - **Meeting demand constraints**
        - **Respecting supplier capacities**
        - **Providing interactive scenario analysis**
        """
        )

        st.subheader("How to Use")
        st.markdown(
            """
        1. **Upload Data**: Upload your supply chain CSV files
        2. **Run Optimization**: Configure parameters and solve
        3. **Analyze Results**: View optimal allocations and costs
        4. **Test Scenarios**: Adjust parameters for what-if analysis
        """
        )

    with col2:
        st.subheader("Key Features")
        st.markdown(
            """
        - **Linear Programming Solver**: Uses PuLP for optimization
        - **Interactive Dashboard**: Real-time parameter adjustment
        - **Visual Analytics**: Charts and tables for insights
        - **Scenario Testing**: Compare different configurations
        - **Cost Breakdown**: Detailed analysis of transportation costs
        """
        )

        # Sample data preview
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame(
            {
                "Supplier": ["S1", "S1", "S2", "S2"],
                "Destination": ["D1", "D2", "D1", "D2"],
                "Cost_per_Unit": [10, 15, 12, 8],
                "Supplier_Capacity": [100, 100, 150, 150],
                "Destination_Demand": [80, 80, 90, 90],
            }
        )
        st.dataframe(sample_data)


def show_data_upload_page():
    st.header("Data Upload")

    # Option to use sample data or upload custom data
    data_option = st.radio(
        "Choose data source:", ["Use Sample Data", "Upload Custom Data"]
    )

    if data_option == "Use Sample Data":
        if st.button("Generate Sample Data"):
            sample_data = DataProcessor.generate_sample_data()
            st.session_state.supply_chain_data = sample_data
            st.success("Sample data generated successfully!")
            st.dataframe(sample_data)

    else:
        st.subheader("Upload Your Supply Chain Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.supply_chain_data = data
                st.success("Data uploaded successfully!")

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data)

                # Data validation
                processor = DataProcessor()
                is_valid, message = processor.validate_data(data)
                if is_valid:
                    st.success(f"Data validation passed: {message}")
                else:
                    st.error(f"Data validation failed: {message}")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # Data requirements information
    with st.expander("Data Format Requirements"):
        st.markdown(
            """
        Your CSV file should contain the following columns:
        - **Supplier**: Supplier identifier (e.g., S1, S2, ...)
        - **Destination**: Destination identifier (e.g., D1, D2, ...)
        - **Cost_per_Unit**: Transportation cost per unit from supplier to destination
        - **Supplier_Capacity**: Maximum capacity of each supplier
        - **Destination_Demand**: Demand requirement at each destination
        """
        )


def show_optimization_page():
    st.header("Optimization Configuration")

    if "supply_chain_data" not in st.session_state:
        st.warning("Please upload data first in the Data Upload page.")
        return

    data = st.session_state.supply_chain_data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Data Summary")
        st.write(f"**Suppliers**: {data['Supplier'].nunique()}")
        st.write(f"**Destinations**: {data['Destination'].nunique()}")
        st.write(f"**Total Routes**: {len(data)}")
        st.write(f"**Total Supply Capacity**: {data['Supplier_Capacity'].sum()}")
        st.write(f"**Total Demand**: {data['Destination_Demand'].sum()}")

    with col2:
        st.subheader("Optimization Parameters")

        # Allow users to adjust demand multiplier for scenario testing
        demand_multiplier = st.slider(
            "Demand Multiplier (for scenario testing)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

        # Allow users to adjust cost multiplier
        cost_multiplier = st.slider(
            "Cost Multiplier (for scenario testing)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

    # Run optimization button
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Solving optimization problem..."):
            try:
                # Apply multipliers for scenario testing
                modified_data = data.copy()
                modified_data["Destination_Demand"] *= demand_multiplier
                modified_data["Cost_per_Unit"] *= cost_multiplier

                # Create and run optimizer
                optimizer = SupplyChainOptimizer()
                results = optimizer.optimize(modified_data)

                st.session_state.optimization_results = results
                st.session_state.modified_data = modified_data

                if results["status"] == "Optimal":
                    st.success("Optimization completed successfully!")

                    # Display key results
                    st.subheader("Optimization Results")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Status", results["status"])
                    with col2:
                        st.metric("Total Cost", f"${results['total_cost']:,.2f}")
                    with col3:
                        st.metric("Routes Used", len(results["allocation"]))

                    # Show allocation table
                    st.subheader("Optimal Allocation")
                    allocation_df = pd.DataFrame(results["allocation"])
                    st.dataframe(allocation_df)

                else:
                    st.error(f"Optimization failed: {results['status']}")

            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")


def show_results_page():
    st.header("Results Analysis")

    if "optimization_results" not in st.session_state:
        st.warning("Please run optimization first.")
        return

    results = st.session_state.optimization_results
    data = st.session_state.modified_data

    # Key metrics
    st.subheader("Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Optimization Status", results["status"])
    with col2:
        st.metric("Total Transportation Cost", f"${results['total_cost']:,.2f}")
    with col3:
        total_shipped = sum([alloc["Quantity"] for alloc in results["allocation"]])
        st.metric("Total Units Shipped", f"{total_shipped:,.0f}")
    with col4:
        avg_cost_per_unit = (
            results["total_cost"] / total_shipped if total_shipped > 0 else 0
        )
        st.metric("Average Cost per Unit", f"${avg_cost_per_unit:.2f}")

    # Allocation visualization
    st.subheader("Shipment Allocation Visualization")

    if results["allocation"]:
        allocation_df = pd.DataFrame(results["allocation"])

        # Create allocation heatmap
        fig_heatmap = create_allocation_heatmap(allocation_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Supply chain network visualization
        st.subheader("Supply Chain Network Flow")
        fig_network = create_supply_chain_graph(allocation_df)
        st.plotly_chart(fig_network, use_container_width=True)

        # Cost breakdown by supplier
        st.subheader("Cost Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            fig_supplier_cost = create_supplier_cost_chart(allocation_df)
            st.plotly_chart(fig_supplier_cost, use_container_width=True)

        with col2:
            fig_destination_flow = create_destination_flow_chart(allocation_df)
            st.plotly_chart(fig_destination_flow, use_container_width=True)

        # Detailed allocation table
        st.subheader("Detailed Allocation Table")
        st.dataframe(allocation_df, use_container_width=True)

        # Download results
        csv_buffer = io.StringIO()
        allocation_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="supply_chain_optimization_results.csv",
            mime="text/csv",
        )


def create_allocation_heatmap(allocation_df):
    """Create a heatmap showing allocation between suppliers and destinations"""
    pivot_data = allocation_df.pivot_table(
        index="Supplier", columns="Destination", values="Quantity", fill_value=0
    )

    fig = px.imshow(
        pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale="Blues",
        title="Shipment Allocation Heatmap",
        labels=dict(x="Destination", y="Supplier", color="Quantity"),
    )

    # Add text annotations
    for i, supplier in enumerate(pivot_data.index):
        for j, destination in enumerate(pivot_data.columns):
            quantity = pivot_data.iloc[i, j]
            if quantity > 0:
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(int(quantity)),
                    showarrow=False,
                    font=dict(
                        color=(
                            "white"
                            if quantity > pivot_data.values.max() / 2
                            else "black"
                        )
                    ),
                )

    return fig


def create_supplier_cost_chart(allocation_df):
    """Create a bar chart showing cost breakdown by supplier"""
    supplier_costs = allocation_df.groupby("Supplier")["Total_Cost"].sum().reset_index()

    fig = px.bar(
        supplier_costs,
        x="Supplier",
        y="Total_Cost",
        title="Transportation Cost by Supplier",
        labels={"Total_Cost": "Total Cost ($)", "Supplier": "Supplier"},
    )

    fig.update_layout(showlegend=False)
    return fig


def create_supply_chain_graph(allocation_df):
    """Create a network graph visualization of the supply chain"""
    import plotly.graph_objects as go

    # Create nodes for suppliers and destinations
    suppliers = allocation_df['Supplier'].unique()
    destinations = allocation_df['Destination'].unique()

    # Create positions for nodes (simple circular layout)
    nodes = list(suppliers) + list(destinations)
    node_positions = {}

    # Position suppliers on the left
    for i, supplier in enumerate(suppliers):
        angle = 2 * 3.14159 * i / len(suppliers)
        node_positions[supplier] = {
            'x': -1,
            'y': 2 * (i - len(suppliers)/2) / len(suppliers)
        }

    # Position destinations on the right
    for i, dest in enumerate(destinations):
        angle = 2 * 3.14159 * i / len(destinations)
        node_positions[dest] = {
            'x': 1,
            'y': 2 * (i - len(destinations)/2) / len(destinations)
        }

    # Create edges (flows between suppliers and destinations)
    edge_x = []
    edge_y = []
    edge_text = []

    for _, row in allocation_df.iterrows():
        if row['Quantity'] > 0:
            supplier = row['Supplier']
            dest = row['Destination']
            quantity = row['Quantity']

            # Add line from supplier to destination
            edge_x.extend([node_positions[supplier]['x'], node_positions[dest]['x'], None])
            edge_y.extend([node_positions[supplier]['y'], node_positions[dest]['y'], None])

            # Add text label in middle of edge
            mid_x = (node_positions[supplier]['x'] + node_positions[dest]['x']) / 2
            mid_y = (node_positions[supplier]['y'] + node_positions[dest]['y']) / 2
            edge_text.append(f"{quantity:.1f} units")

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='blue'),
        hoverinfo='text',
        text=edge_text,
        name='Flows'
    ))

    # Add nodes
    node_x = [pos['x'] for pos in node_positions.values()]
    node_y = [pos['y'] for pos in node_positions.values()]
    node_text = nodes
    node_colors = ['lightblue'] * len(suppliers) + ['lightgreen'] * len(destinations)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=40, color=node_colors, line=dict(width=2, color='black')),
        text=node_text,
        textposition="middle center",
        hoverinfo='text',
        name='Nodes'
    ))

    # Update layout
    return fig


def create_destination_flow_chart(allocation_df):
    """Create a chart showing flow to destinations"""
    destination_flow = (
        allocation_df.groupby("Destination")["Quantity"].sum().reset_index()
    )

    fig = px.pie(
        destination_flow,
        values="Quantity",
        names="Destination",
        title="Shipment Distribution by Destination",
    )

    return fig


if __name__ == "__main__":
    main()
