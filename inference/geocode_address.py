import requests
import os

def geocode_address_locationiq(address: str, api_key: str):
    url = "https://us1.locationiq.com/v1/search.php"
    params = {
        "key": api_key,
        "q": address,
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"Could not geocode address: {address}")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

# # Geocode test addresses
# print("LocationIQ results:")
# print("Pickup:", geocode_address_locationiq("150 5th Ave, New York, NY 10118", api_key))
# print("Dropoff:", geocode_address_locationiq("1560 Broadway, New York, NY 10036, USA", api_key))


def geocode_address_nominatim(address: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "TaxiFareApp/1.0 (contact@example.com)"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"Could not geocode address: {address}")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon