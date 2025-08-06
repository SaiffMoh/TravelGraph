import os
import streamlit as st
import json
from dotenv import load_dotenv
from state_definitions import FlightAssistantState
from graph_workflow import create_flight_assistant_graph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import traceback

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

# Main Streamlit application
st.set_page_config(page_title="AI Flight Assistant", layout="wide")
st.title("AI Flight Assistant")
st.write("Powered by LangGraph, Amadeus, and OpenAI")

# Initialize Streamlit session states
if "flight_state" not in st.session_state:
    st.session_state.flight_state = FlightAssistantState()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI flight assistant. I can help you find and analyze flight options. Tell me where you'd like to go and when you'd like to travel!"
        }
    ]
# Create the workflow graph
if "workflow" not in st.session_state:
    st.session_state.workflow = create_flight_assistant_graph()

# Display flight search results in a structured format
def display_flight_results(state):
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
# := assigns the return value while also checking if it's not empty
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

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your input..."):

            try:
                # Convert full chat history into a single string
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

                # Inject the prompt with the template to guide question flow
                full_prompt = question_prompt.format(chat_history=chat_history)

                # Get assistant's response using the full prompt
                response = llm.predict(text=full_prompt)

                # Show and log assistant response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.flight_state.messages.append({
                    "role": "assistant", "content": response
                })

                # Parse and update state from user's last input
                user_input = prompt.lower()

                if not st.session_state.flight_state.departure_date and "202" in user_input:
                    st.session_state.flight_state.departure_date = user_input.strip()

                elif not st.session_state.flight_state.origin and any(c in user_input for c in ["from", "origin", "cairo", "new york"]):
                    st.session_state.flight_state.origin = user_input.strip()

                elif not st.session_state.flight_state.destination and any(c in user_input for c in ["to", "destination", "dubai", "london"]):
                    st.session_state.flight_state.destination = user_input.strip()

                elif not st.session_state.flight_state.cabin_class and any(c in user_input for c in ["economy", "business", "first"]):
                    st.session_state.flight_state.cabin_class = user_input.strip().capitalize()

                elif not st.session_state.flight_state.trip_type and any(c in user_input for c in ["one way", "round", "round trip"]):
                    st.session_state.flight_state.trip_type = "round_trip" if "round" in user_input else "one_way"

                elif st.session_state.flight_state.trip_type == "round_trip" and not st.session_state.flight_state.duration:
                    # Basic fallback if user enters duration like "5 days"
                    import re
                    match = re.search(r"(\d+)\s*days?", user_input)
                    if match:
                        st.session_state.flight_state.duration = int(match.group(1))

                # Check if all required fields are available to trigger the LangGraph flow
                state = st.session_state.flight_state
                required_fields_one_way = [state.departure_date, state.origin, state.destination, state.cabin_class, "one_way"]
                required_fields_round_trip = [state.departure_date, state.origin, state.destination, state.cabin_class, "round_trip", state.duration]
                print(required_fields_one_way)
                print(required_fields_round_trip)

                if (
                    (state.trip_type == "one_way" and all(required_fields_one_way)) or
                    (state.trip_type == "round_trip" and all(required_fields_round_trip))
                ):
                    st.success("All inputs collected! Starting flight search...")

                    # Run LangGraph workflow (this starts the defined node-based flow)
                    result_state = st.session_state.workflow.invoke(state)
                    st.session_state.flight_state = result_state

                    # Display results
                    display_flight_results(result_state)

                else:
                    st.info("Waiting for more info...")

            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}. Please try again or rephrase your request."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                traceback.print_exc()


# Sidebar with current search parameters
with st.sidebar:
    st.header("Current Search")
    state = st.session_state.flight_state
    # print(state)
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