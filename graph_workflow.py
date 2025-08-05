from langgraph.graph import StateGraph, END
from state_definitions import FlightAssistantState
from graph_nodes import (
    collect_user_input_node,
    fetch_amadeus_token_node,
    call_flight_offers_api_node,
    analyze_offers_with_llm_node,
    display_results_node
)

def create_flight_assistant_graph():
    """Create the LangGraph workflow for the flight assistant"""

    # Create the graph
    workflow = StateGraph(FlightAssistantState)

    # Add nodes
    workflow.add_node("collect_input", collect_user_input_node)
    workflow.add_node("fetch_token", fetch_amadeus_token_node)
    workflow.add_node("search_flights", call_flight_offers_api_node)
    workflow.add_node("analyze_results", analyze_offers_with_llm_node)
    workflow.add_node("display_results", display_results_node)

    # Define the workflow edges
    workflow.add_edge("collect_input", "fetch_token")
    workflow.add_edge("fetch_token", "search_flights")
    workflow.add_edge("search_flights", "analyze_results")
    workflow.add_edge("analyze_results", "display_results")
    workflow.add_edge("display_results", END)

    # Set entry point
    workflow.set_entry_point("collect_input")

    return workflow.compile()
