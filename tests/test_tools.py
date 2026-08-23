"""
test_tools.py — Unit tests for the travel assistant tool implementations.

All external HTTP calls are mocked with unittest.mock.patch so these tests
run offline without consuming any API quota.

Test coverage:
  _get_coordinates()     — geocoding helper (happy path + unknown city)
  get_current_weather()  — full tool (happy path, unknown city, network error)
"""
from unittest.mock import MagicMock, patch

import pytest

from app.tools.weather import WeatherResult, _get_coordinates, get_current_weather


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_geo_response(results: list) -> MagicMock:
    """Build a mock requests.Response for the geocoding API."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"results": results}
    return mock


def _make_weather_response(temp_c: float, windspeed: float, code: int, is_day: int) -> MagicMock:
    """Build a mock requests.Response for the Open-Meteo forecast API."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "current": {
            "temperature_2m": temp_c,
            "windspeed_10m": windspeed,
            "weathercode": code,
            "is_day": is_day,
        }
    }
    return mock


# ---------------------------------------------------------------------------
# Tests — _get_coordinates
# ---------------------------------------------------------------------------
class TestGetCoordinates:
    """Unit tests for the internal geocoding helper."""

    def test_returns_coordinates_for_known_city(self):
        """Happy path: a valid city returns (lat, lon, name, country)."""
        mock_response = _make_geo_response([
            {"latitude": 48.8566, "longitude": 2.3522, "name": "Paris", "country": "France"}
        ])
        with patch("app.tools.weather.requests.get", return_value=mock_response):
            lat, lon, city, country = _get_coordinates("Paris")

        assert lat == 48.8566
        assert lon == 2.3522
        assert city == "Paris"
        assert country == "France"

    def test_raises_value_error_for_unknown_city(self):
        """When the API returns no results, ValueError should be raised."""
        mock_response = _make_geo_response([])
        with patch("app.tools.weather.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="could not be found"):
                _get_coordinates("ZZZNonexistentCityXXX")

    def test_uses_city_name_as_fallback_when_name_missing(self):
        """If the API result omits the 'name' key, the input city string is used."""
        mock_response = _make_geo_response([
            {"latitude": 35.6895, "longitude": 139.6917, "country": "Japan"}
            # 'name' key intentionally absent
        ])
        with patch("app.tools.weather.requests.get", return_value=mock_response):
            _, _, city, _ = _get_coordinates("Tokyo")

        assert city == "Tokyo"


# ---------------------------------------------------------------------------
# Tests — get_current_weather (the @tool function)
# ---------------------------------------------------------------------------
class TestGetCurrentWeather:
    """Unit tests for the get_current_weather LangChain tool."""

    def test_returns_formatted_weather_string_for_valid_city(self):
        """Happy path: valid city returns a string with temperature and conditions."""
        geo_response = _make_geo_response([
            {"latitude": 48.85, "longitude": 2.35, "name": "Paris", "country": "France"}
        ])
        weather_response = _make_weather_response(
            temp_c=18.5, windspeed=12.0, code=2, is_day=1
        )
        with patch("app.tools.weather.requests.get", side_effect=[geo_response, weather_response]):
            result = get_current_weather.invoke({"city": "Paris"})

        assert "Paris" in result
        assert "France" in result
        assert "18.5" in result
        assert "°C" in result
        assert "°F" in result
        assert "km/h" in result

    def test_temperature_fahrenheit_conversion_is_correct(self):
        """18°C should convert to 64.4°F."""
        geo_response = _make_geo_response([
            {"latitude": 51.5, "longitude": -0.12, "name": "London", "country": "United Kingdom"}
        ])
        weather_response = _make_weather_response(
            temp_c=18.0, windspeed=15.0, code=61, is_day=1
        )
        with patch("app.tools.weather.requests.get", side_effect=[geo_response, weather_response]):
            result = get_current_weather.invoke({"city": "London"})

        assert "64.4" in result

    def test_returns_error_string_for_unknown_city(self):
        """An unknown city should return a descriptive error string, not raise."""
        geo_response = _make_geo_response([])
        with patch("app.tools.weather.requests.get", return_value=geo_response):
            result = get_current_weather.invoke({"city": "FakeCity99999"})

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention the city couldn't be found
        assert "found" in result.lower() or "could not" in result.lower()

    def test_returns_error_string_on_network_failure(self):
        """A network error during geocoding should return a descriptive error string."""
        import requests as req
        with patch("app.tools.weather.requests.get", side_effect=req.RequestException("timeout")):
            result = get_current_weather.invoke({"city": "Tokyo"})

        assert isinstance(result, str)
        assert "Could not" in result

    def test_nighttime_flag_included_in_output(self):
        """Nighttime conditions should appear in the output string."""
        geo_response = _make_geo_response([
            {"latitude": 35.68, "longitude": 139.69, "name": "Tokyo", "country": "Japan"}
        ])
        weather_response = _make_weather_response(
            temp_c=15.0, windspeed=8.0, code=1, is_day=0  # is_day=0 means night
        )
        with patch("app.tools.weather.requests.get", side_effect=[geo_response, weather_response]):
            result = get_current_weather.invoke({"city": "Tokyo"})

        assert "nighttime" in result

    def test_weather_result_model_validates_correctly(self):
        """WeatherResult Pydantic model should accept valid data without error."""
        wr = WeatherResult(
            city="Sydney",
            country="Australia",
            temperature_celsius=25.0,
            temperature_fahrenheit=77.0,
            windspeed_kmh=20.0,
            weather_description="Clear sky",
            is_day=True,
        )
        assert wr.city == "Sydney"
        assert wr.temperature_fahrenheit == 77.0
