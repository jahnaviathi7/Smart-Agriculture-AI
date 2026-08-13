import requests
import pandas as pd


# ============================================================
# WEATHER CODE DESCRIPTION
# ============================================================

def weather_description(code):

    descriptions = {
        0: "☀️ Clear sky",
        1: "🌤️ Mainly clear",
        2: "⛅ Partly cloudy",
        3: "☁️ Overcast",
        45: "🌫️ Fog",
        48: "🌫️ Depositing rime fog",
        51: "🌦️ Light drizzle",
        53: "🌦️ Moderate drizzle",
        55: "🌧️ Dense drizzle",
        61: "🌦️ Slight rain",
        63: "🌧️ Moderate rain",
        65: "🌧️ Heavy rain",
        71: "🌨️ Slight snow",
        73: "🌨️ Moderate snow",
        75: "❄️ Heavy snow",
        80: "🌦️ Slight rain showers",
        81: "🌧️ Moderate rain showers",
        82: "⛈️ Violent rain showers",
        95: "⛈️ Thunderstorm",
        96: "⛈️ Thunderstorm with slight hail",
        99: "⛈️ Thunderstorm with heavy hail"
    }

    return descriptions.get(code, "🌤️ Unknown")


# ============================================================
# LOCATION SEARCH
# ============================================================

def get_location(location):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if "results" not in data or not data["results"]:
        return None

    result = data["results"][0]

    return {
        "name": result["name"],
        "country": result.get("country", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"]
    }


# ============================================================
# WEATHER DATA
# ============================================================

def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation_probability,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "rain_sum,"
            "precipitation_probability_max,"
            "weather_code"
        ),

        "forecast_days": 7,
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# AGRICULTURE ADVISORY
# ============================================================

def agriculture_advisory(weather):

    current = weather["current"]

    temperature = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    precipitation = current["precipitation"]
    wind = current["wind_speed_10m"]

    advice = []

    if precipitation >= 5:
        advice.append(
            "🌧️ Rain is occurring. Avoid unnecessary irrigation."
        )

    elif temperature >= 35 and humidity < 40:
        advice.append(
            "💧 High heat and low humidity. Monitor soil moisture closely."
        )

    else:
        advice.append(
            "💧 Weather conditions do not indicate immediate irrigation stress."
        )

    if temperature >= 38:
        advice.append(
            "🌡️ Very high temperature. Consider heat protection for crops."
        )

    if humidity >= 85:
        advice.append(
            "💦 High humidity may increase fungal disease risk."
        )

    if wind >= 30:
        advice.append(
            "💨 Strong winds detected. Avoid spraying pesticides/fertilizers."
        )

    return advice


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    location = input(
        "Enter city/location: "
    ).strip()

    location_data = get_location(location)

    if location_data is None:

        print("\n❌ Location not found.")

    else:

        print(
            f"\n📍 Location: "
            f"{location_data['name']}, "
            f"{location_data['country']}"
        )

        weather = get_weather(
            location_data["latitude"],
            location_data["longitude"]
        )

        current = weather["current"]

        print("\n===================================")
        print("🌦️ WEATHER ANALYSIS")
        print("===================================")

        print(
            f"Temperature     : "
            f"{current['temperature_2m']} °C"
        )

        print(
            f"Humidity        : "
            f"{current['relative_humidity_2m']}%"
        )

        print(
            f"Precipitation   : "
            f"{current['precipitation']} mm"
        )

        print(
            f"Wind Speed      : "
            f"{current['wind_speed_10m']} km/h"
        )

        print(
            f"Condition       : "
            f"{weather_description(current['weather_code'])}"
        )

        print("\n🌾 FARMING ADVISORY")

        for advice in agriculture_advisory(weather):
            print(advice)

        print("\n===================================")