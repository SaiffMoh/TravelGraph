import os
import time
import requests
import openai
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
from state_definitions import FlightAssistantState
from utility_functions import format_flight_offers_body, extract_flight_info
import json

# Load environment variables from .env file
load_dotenv()

def collect_user_input_node(state: Dict) -> Dict:
    """Collect and validate user input for flight search"""
    state = FlightAssistantState.from_dict(state)

    if not state.messages:
        state.current_step = "greeting"
        return state.to_dict()

    latest_message = state.messages[-1]['content'] if state.messages else ""

    # Extract flight information from the latest message
    extracted_info = extract_flight_info(latest_message, state)

    # Update state with extracted information
    for key, value in extracted_info.items():
        setattr(state, key, value)

    # Check what information is still missing
    required_fields = ['origin', 'destination', 'departure_date']
    missing = []

    for field in required_fields:
        if not getattr(state, field):
            missing.append(field)

    # If round trip, also need duration
    if state.trip_type == "round_trip" and not state.duration:
        missing.append('duration')

    state.missing_fields = missing

    if missing:
        state.current_step = "collect_missing"
    else:
        state.current_step = "ready_to_search"

    return state.to_dict()

def fetch_amadeus_token_node(state: Dict) -> Dict:
    """Fetch or refresh Amadeus API token"""
    state = FlightAssistantState.from_dict(state)

    # Check if we have a valid token
    if state.amadeus_token and state.token_expires_at:
        if time.time() < state.token_expires_at - 300:  # 5 min buffer
            return state.to_dict()

    # Fetch new token
    try:
        client_id = os.getenv("AMADEUS_CLIENT_ID")
        client_secret = os.getenv("AMADEUS_CLIENT_SECRET")

        if not client_id or not client_secret:
            state.last_error = "Amadeus API credentials not configured"
            return state.to_dict()

        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }

        response = requests.post(token_url, data=token_data)
        response.raise_for_status()

        token_info = response.json()
        state.amadeus_token = token_info["access_token"]
        state.token_expires_at = time.time() + token_info["expires_in"]
        state.last_error = None

    except Exception as e:
        state.last_error = f"Failed to fetch Amadeus token: {str(e)}"

    return state.to_dict()

def call_flight_offers_api_node(state: Dict) -> Dict:
    """Call the Amadeus Flight Offers API"""
    state = FlightAssistantState.from_dict(state)

    if not state.amadeus_token:
        state.last_error = "No valid Amadeus token available"
        return state.to_dict()

    try:
        # Prepare the API request
        api_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {
            "Authorization": f"Bearer {state.amadeus_token}",
            "Content-Type": "application/json"
        }

        # Format the request body
        request_body = format_flight_offers_body(
            currency_code=state.currency,
            origin_location_code=state.origin,
            destination_location_code=state.destination,
            departure_date=state.departure_date,
            trip_type=state.trip_type,
            duration=state.duration,
            cabin=state.cabin_class,
            max_flight_offers=3
        )

        # Make the API call
        response = requests.post(api_url, headers=headers, json=request_body)
        response.raise_for_status()

        flight_data = response.json()
        state.flight_offers = flight_data.get("data", [])
        state.last_error = None

        if not state.flight_offers:
            state.last_error = "No flight offers found for your search criteria"

    except requests.exceptions.RequestException as e:
        state.last_error = f"API request failed: {str(e)}"
    except Exception as e:
        state.last_error = f"Error calling flight offers API: {str(e)}"

    return state.to_dict()

def analyze_offers_with_llm_node(state: Dict) -> Dict:
    """Analyze flight offers using OpenAI LLM"""
    state = FlightAssistantState.from_dict(state)

    if not state.flight_offers:
        state.llm_analysis = "No flight offers available to analyze."
        return state.to_dict()

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            state.last_error = "OpenAI API key not configured"
            return state.to_dict()

        client = openai.OpenAI(api_key=openai_api_key)

        # Prepare flight data for analysis
        flight_summary = []
        for i, offer in enumerate(state.flight_offers[:3], 1):
            try:
                price = offer.get("price", {}).get("total", "N/A")
                currency = offer.get("price", {}).get("currency", state.currency)

                itineraries = offer.get("itineraries", [])
                segments_info = []

                for itinerary in itineraries:
                    segments = itinerary.get("segments", [])
                    duration = itinerary.get("duration", "N/A")

                    for segment in segments:
                        departure = segment.get("departure", {})
                        arrival = segment.get("arrival", {})
                        carrier = segment.get("carrierCode", "N/A")
                        flight_number = segment.get("number", "N/A")

                        segments_info.append({
                            "carrier": carrier,
                            "flight": f"{carrier}{flight_number}",
                            "from": departure.get("iataCode", "N/A"),
                            "to": arrival.get("iataCode", "N/A"),
                            "departure_time": departure.get("at", "N/A"),
                            "arrival_time": arrival.get("at", "N/A")
                        })

                flight_summary.append({
                    "option": i,
                    "price": f"{price} {currency}",
                    "segments": segments_info,
                    "total_duration": duration
                })
            except Exception as e:
                flight_summary.append({
                    "option": i,
                    "error": f"Error parsing flight {i}: {str(e)}"
                })

        # Create prompt for LLM analysis
        prompt = f"""
        Analyze the following flight options for a trip from {state.origin} to {state.destination} on {state.departure_date}:
        {json.dumps(flight_summary, indent=2)}
        Trip type: {state.trip_type}
        Cabin class: {state.cabin_class}

        Please provide a friendly, conversational analysis that:
        1. Highlights the best option based on common preferences (price, duration, convenience)
        2. Explains the trade-offs between options
        3. Mentions any notable features (direct flights, short layovers, etc.)
        4. Uses a warm, helpful tone as if you're a travel advisor

        Keep it concise but informative, around 150-200 words.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful travel advisor who analyzes flight options and provides friendly, practical advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )

        state.llm_analysis = response.choices[0].message.content
        state.last_error = None

    except Exception as e:
        state.last_error = f"Error analyzing flights with LLM: {str(e)}"
        state.llm_analysis = "I couldn't analyze the flight options right now, but I found some flights for you!"

    return state.to_dict()

def display_results_node(state: Dict) -> Dict:
    """Prepare results for display"""
    state = FlightAssistantState.from_dict(state)
    state.current_step = "results_ready"
    return state.to_dict()
