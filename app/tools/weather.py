"""
weather.py — Real-time weather tool for the travel assistant.

Uses the Open-Meteo API (https://open-meteo.com/) for weather data and the
Open-Meteo Geocoding API to resolve city names to coordinates.

Design decisions:
- Open-Meteo is 100% free and requires no API key, removing all setup friction.
- City names are resolved via geocoding so users can type natural names
  ("Tokyo", "New York", "São Paulo") instead of coordinates.
- The internal result is validated as a Pydantic model before being
  serialised to a plain-text string. This catches bad API responses early
  and makes the tool independently testable with structured assertions.
- Temperature is returned in both Celsius and Fahrenheit to serve
  international users regardless of their preference.
- The @tool decorator auto-generates a JSON schema from the function
  signature and docstring. The LLM reads the docstring to decide when
  and how to invoke the tool, so clarity here directly improves accuracy.
"""
import requests
from langchain_core.tools import tool
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# WMO Weather interpretation codes
# Reference: https://open-meteo.com/en/docs#weathervariables
# ---------------------------------------------------------------------------
WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------
class WeatherResult(BaseModel):
    """Structured result returned by the weather API before serialisation."""

    city: str
    country: str
    temperature_celsius: float
    temperature_fahrenheit: float
    windspeed_kmh: float
    weather_description: str
    is_day: bool


# ---------------------------------------------------------------------------
# Helper — geocoding
# ---------------------------------------------------------------------------
def _get_coordinates(city: str) -> tuple[float, float, str, str]:
    """Resolve a city name to (latitude, longitude, city_name, country).

    Uses the Open-Meteo Geocoding API which is free and requires no key.

    Args:
        city: Human-readable city name, e.g. "Paris" or "New York".

    Returns:
        A tuple of (latitude, longitude, resolved_city_name, country_name).

    Raises:
        ValueError: If the city cannot be found.
        requests.RequestException: If the network request fails.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(
        url,
        params={"name": city, "count": 1, "language": "en"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
        raise ValueError(
            f"City '{city}' could not be found. "
            "Try a different spelling or a nearby major city."
        )

    result = data["results"][0]
    return (
        result["latitude"],
        result["longitude"],
        result.get("name", city),
        result.get("country", "Unknown"),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@tool
def get_current_weather(city: str) -> str:
    """Get the current weather conditions for any city in the world.

    Use this tool whenever the user asks about:
    - Current weather at a travel destination
    - Temperature, rain, snow, or wind conditions
    - Whether it is a good time to visit based on weather
    - What to pack for a trip

    Args:
        city: The name of the city to look up, e.g. "Paris", "Tokyo",
              "New York", "Sydney".

    Returns:
        A plain-text summary of current weather including sky conditions,
        temperature in both Celsius and Fahrenheit, and wind speed.
        Returns a descriptive error message if the city cannot be found
        or the weather service is unavailable.
    """
    # Step 1 — resolve city name to coordinates
    try:
        lat, lon, resolved_city, country = _get_coordinates(city)
    except ValueError as exc:
        return str(exc)
    except requests.RequestException as exc:
        return f"Could not look up coordinates for '{city}': {exc}"

    # Step 2 — fetch current weather from Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,windspeed_10m,weathercode,is_day",
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Could not fetch weather data for '{resolved_city}': {exc}"

    # Step 3 — parse and validate response
    current = data["current"]
    temp_c = current["temperature_2m"]
    temp_f = round(temp_c * 9 / 5 + 32, 1)

    result = WeatherResult(
        city=resolved_city,
        country=country,
        temperature_celsius=temp_c,
        temperature_fahrenheit=temp_f,
        windspeed_kmh=current["windspeed_10m"],
        weather_description=WMO_DESCRIPTIONS.get(current["weathercode"], "Unknown conditions"),
        is_day=bool(current["is_day"]),
    )

    # Step 4 — serialise to human-readable string for the LLM
    time_of_day = "daytime" if result.is_day else "nighttime"
    return (
        f"Current weather in {result.city}, {result.country}:\n"
        f"  Conditions : {result.weather_description} ({time_of_day})\n"
        f"  Temperature: {result.temperature_celsius}°C "
        f"({result.temperature_fahrenheit}°F)\n"
        f"  Wind speed : {result.windspeed_kmh} km/h"
    )
