import requests

BASE_URL = "https://example.com"

def fetch_user(user_id: str) -> dict:
    """Return {'id': ..., 'name': ...} for a user."""
    url = f"{BASE_URL}/users/{user_id}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    data = {
        'id' : data['id'],
        'name' : data['name']
    }
    return data
