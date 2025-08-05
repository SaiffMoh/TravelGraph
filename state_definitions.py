from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

@dataclass
class FlightAssistantState:
    """Central state for the flight assistant workflow"""
    # User search parameters
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    trip_type: str = "one_way"  # "one_way" or "round_trip"
    duration: Optional[int] = None
    travelers: int = 1
    cabin_class: str = "ECONOMY"
    currency: str = "USD"

    # API tokens and data
    amadeus_token: Optional[str] = None
    token_expires_at: Optional[float] = None
    flight_offers: List[Dict[str, Any]] = field(default_factory=list)

    # LLM analysis
    llm_analysis: Optional[str] = None

    # Conversation context
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_step: str = "greeting"
    missing_fields: List[str] = field(default_factory=list)
    conversation_context: str = ""

    # Error handling
    last_error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert the dataclass to a dictionary"""
        return {
            'origin': self.origin,
            'destination': self.destination,
            'departure_date': self.departure_date,
            'return_date': self.return_date,
            'trip_type': self.trip_type,
            'duration': self.duration,
            'travelers': self.travelers,
            'cabin_class': self.cabin_class,
            'currency': self.currency,
            'amadeus_token': self.amadeus_token,
            'token_expires_at': self.token_expires_at,
            'flight_offers': self.flight_offers,
            'llm_analysis': self.llm_analysis,
            'messages': self.messages,
            'current_step': self.current_step,
            'missing_fields': self.missing_fields,
            'conversation_context': self.conversation_context,
            'last_error': self.last_error
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FlightAssistantState':
        """Create a FlightAssistantState from a dictionary"""
        return cls(
            origin=data.get('origin'),
            destination=data.get('destination'),
            departure_date=data.get('departure_date'),
            return_date=data.get('return_date'),
            trip_type=data.get('trip_type', 'one_way'),
            duration=data.get('duration'),
            travelers=data.get('travelers', 1),
            cabin_class=data.get('cabin_class', 'ECONOMY'),
            currency=data.get('currency', 'USD'),
            amadeus_token=data.get('amadeus_token'),
            token_expires_at=data.get('token_expires_at'),
            flight_offers=data.get('flight_offers', []),
            llm_analysis=data.get('llm_analysis'),
            messages=data.get('messages', []),
            current_step=data.get('current_step', 'greeting'),
            missing_fields=data.get('missing_fields', []),
            conversation_context=data.get('conversation_context', ''),
            last_error=data.get('last_error')
        )
