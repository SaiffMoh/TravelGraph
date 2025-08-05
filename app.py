import os
import streamlit as st
import json
from dotenv import load_dotenv
from state_definitions import FlightAssistantState
from graph_workflow import create_flight_assistant_graph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM for the chat
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Define the prompt template for the chat
question_prompt = ChatPromptTemplate.from_template(
    """
You are a friendly travel assistant helping a user book a flight. Based on the chat history below, ask the next most relevant question to collect missing information needed to search for flight offers. Only ask one question at a time. Be conversational, warm, and clear. Use this order for questions:
1. Departure date
2. Origin (city or airport)
3. Destination (city or airport)
4. Cabin class (Economy, Business, etc.)
5. Trip type (one way or round trip)
6. If round trip, ask for duration of stay in days

Required fields for one way:
- departure_date
- origin_location_code
- destination_location_code
- cabin
- trip_type (must be 'one_way')

Required fields for round trip:
- departure_date
- origin_location_code
- destination_location_code
- cabin
- trip_type (must be 'round_trip')
- duration (number of days between outbound and return)

After each answer, check if all required fields for the selected trip type are filled.

ASSUMPTIONS FOR DATES:
- If the user omits the year, assume the current year, unless the month is before the current month, in which case assume next year.
- If the user omits the month and year, assume the current month and year, unless the day is before today, in which case assume next month (and next year if month is December).
- Always return the date in YYYY-MM-DD format.

Chat history:
{chat_history}
"""
)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "flight_state" not in st.session_state:
        st.session_state.flight_state = FlightAssistantState()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI flight assistant. I can help you find and analyze flight options. Tell me where you'd like to go and when you'd like to travel!"
            }
        ]

def display_flight_results(state):
    """Display flight search results in a structured format"""
    if state.last_error:
        st.error(f"Error: {state.last_error}")
        return

    if not state.flight_offers:
        st.warning("No flights found for your search criteria.")
        return

    st.subheader("Flight Options Found")

    # Display LLM analysis first
    if state.llm_analysis:
        st.info(state.llm_analysis)

    # Display individual flight options
    for i, offer in enumerate(state.flight_offers[:3], 1):
        with st.expander(f"Flight Option {i}", expanded=(i == 1)):
            try:
                price = offer.get("price", {})
                st.metric("Price", f"{price.get('total', 'N/A')} {price.get('currency', 'USD')}")

                itineraries = offer.get("itineraries", [])
                for j, itinerary in enumerate(itineraries):
                    st.write(f"**{'Outbound' if j == 0 else 'Return'} Journey:**")

                    segments = itinerary.get("segments", [])
                    duration = itinerary.get("duration", "N/A")

                    st.write(f"Total Duration: {duration}")

                    for k, segment in enumerate(segments):
                        departure = segment.get("departure", {})
                        arrival = segment.get("arrival", {})
                        carrier = segment.get("carrierCode", "N/A")
                        flight_number = segment.get("number", "N/A")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**{carrier}{flight_number}**")
                            st.write(f"From: {departure.get('iataCode', 'N/A')}")
                        with col2:
                            st.write(f"Depart: {departure.get('at', 'N/A')}")
                            st.write(f"Arrive: {arrival.get('at', 'N/A')}")
                        with col3:
                            st.write(f"To: {arrival.get('iataCode', 'N/A')}")

                        if k < len(segments) - 1:
                            st.write("↓ *Connection*")

                    if j < len(itineraries) - 1:
                        st.write("---")

            except Exception as e:
                st.error(f"Error displaying flight option {i}: {str(e)}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🛫 AI Flight Assistant",
        page_icon="✈️",
        layout="wide"
    )

    st.title("🛫 AI Flight Assistant")
    st.write("Powered by LangGraph, Amadeus, and OpenAI")

    # Initialize session state
    initialize_session_state()

    # Create the workflow graph
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_flight_assistant_graph()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Tell me about your travel plans..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Add message to flight state
        st.session_state.flight_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Process with LangGraph workflow
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    # Generate response based on current step using LLM
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = llm.predict(chat_history=chat_history)

                    # Display and store response
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Update the state based on the response
                    if "departure date" in response.lower():
                        st.session_state.flight_state.current_step = "collect_departure_date"
                    elif "origin" in response.lower():
                        st.session_state.flight_state.current_step = "collect_origin"
                    elif "destination" in response.lower():
                        st.session_state.flight_state.current_step = "collect_destination"
                    elif "cabin class" in response.lower():
                        st.session_state.flight_state.current_step = "collect_cabin_class"
                    elif "trip type" in response.lower():
                        st.session_state.flight_state.current_step = "collect_trip_type"
                    elif "duration" in response.lower():
                        st.session_state.flight_state.current_step = "collect_duration"
                    else:
                        st.session_state.flight_state.current_step = "ready_to_search"

                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again or rephrase your request."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Sidebar with current search parameters
    with st.sidebar:
        st.header("Current Search")
        state = st.session_state.flight_state

        if state.origin:
            st.write(f"**From:** {state.origin}")
        if state.destination:
            st.write(f"**To:** {state.destination}")
        if state.departure_date:
            st.write(f"**Departure:** {state.departure_date}")
        if state.trip_type == "round_trip" and state.duration:
            st.write(f"**Duration:** {state.duration} days")

        st.write(f"**Trip Type:** {state.trip_type.replace('_', ' ').title()}")
        st.write(f"**Cabin:** {state.cabin_class}")

        if st.button("Clear Search"):
            st.session_state.flight_state = FlightAssistantState()
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your AI flight assistant. I can help you find and analyze flight options. Tell me where you'd like to go and when you'd like to travel!"
                }
            ]
            st.rerun()

if __name__ == "__main__":
    main()
