import json
from unittest.mock import patch, Mock
import api_client

def test_fetch_user_parses_response(mocker):
    fake_json = {"id": "u_123", "name": "Ada", "email": "ada@example.com"}
    # first let's build the response 
    payload = mocker.Mock()
    payload.json.return_value = fake_json
    payload.raise_for_status.return_value = None 

    # then let's indicate where this call to a network is going to land
    mock_request_get = mocker.patch("api_client.requests.get", return_value=payload)

    # call the actual function 
    user_data = api_client.fetch_user("u_123")
    mock_request_get.assert_called_once_with("https://example.com/users/u_123", timeout=5)
    assert user_data == {'id': fake_json['id'], 'name': fake_json['name']}