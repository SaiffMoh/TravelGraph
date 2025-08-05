import datetime
import re
from typing import Dict, Any, List, Optional
from state_definitions import FlightAssistantState

def format_flight_offers_body(
    currency_code: str,
    origin_location_code: str,
    destination_location_code: str,
    departure_date: str,
    departure_time: str = "10:00:00",
    traveler_type: str = "ADULT",
    max_flight_offers: int = 3,
    cabin: str = "ECONOMY",
    coverage: str = "MOST_SEGMENTS",
    sources: Optional[List[str]] = None,
    trip_type: str = "one_way",
    duration: Optional[int] = None
) -> Dict[str, Any]:
    """Format the Amadeus API request body"""
    if sources is None:
        sources = ["GDS"]

    origin_destinations = [
        {
            "id": "1",
            "originLocationCode": origin_location_code,
            "destinationLocationCode": destination_location_code,
            "departureDateTimeRange": {
                "date": departure_date,
                "time": departure_time
            }
        }
    ]

    if trip_type == "round_trip" and duration is not None:
        dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d")
        return_date = (dep_date + datetime.timedelta(days=int(duration))).strftime("%Y-%m-%d")
        origin_destinations.append({
            "id": "2",
            "originLocationCode": destination_location_code,
            "destinationLocationCode": origin_location_code,
            "departureDateTimeRange": {
                "date": return_date,
                "time": "10:00:00"
            }
        })

    return {
        "currencyCode": currency_code,
        "originDestinations": origin_destinations,
        "travelers": [
            {
                "id": "1",
                "travelerType": traveler_type
            }
        ],
        "sources": sources,
        "searchCriteria": {
            "maxFlightOffers": max_flight_offers,
            "flightFilters": {
                "cabinRestrictions": [
                    {
                        "cabin": cabin,
                        "coverage": coverage,
                        "originDestinationIds": [od["id"] for od in origin_destinations]
                    }
                ]
            }
        }
    }

def extract_flight_info(user_input: str, current_state: 'FlightAssistantState') -> Dict[str, Any]:
    """Extract flight information from user input using pattern matching and NLP"""
    info = {}

    # Extract airport codes (3 letters, uppercase)
    airport_codes = re.findall(r'\b[A-Z]{3}\b', user_input.upper())
    if len(airport_codes) >= 2:
        info['origin'] = airport_codes[0]
        info['destination'] = airport_codes[1]

    # Extract dates (various formats)
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, user_input)
        if dates:
            try:
                # Try to parse the first date as departure
                if '/' in dates[0]:
                    parsed_date = datetime.datetime.strptime(dates[0], "%m/%d/%Y")
                elif '-' in dates[0] and len(dates[0].split('-')[0]) <= 2:
                    parsed_date = datetime.datetime.strptime(dates[0], "%m-%d-%Y")
                else:
                    parsed_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")

                info['departure_date'] = parsed_date.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    # Extract trip type
    if any(word in user_input.lower() for word in ['round trip', 'return', 'round-trip']):
        info['trip_type'] = 'round_trip'
    elif any(word in user_input.lower() for word in ['one way', 'one-way']):
        info['trip_type'] = 'one_way'

    # Extract duration for round trips
    duration_match = re.search(r'(\d+)\s*day', user_input.lower())
    if duration_match:
        info['duration'] = int(duration_match.group(1))

    # Extract cabin class
    if 'business' in user_input.lower():
        info['cabin_class'] = 'BUSINESS'
    elif 'first' in user_input.lower():
        info['cabin_class'] = 'FIRST'
    elif 'economy' in user_input.lower():
        info['cabin_class'] = 'ECONOMY'

    return info
