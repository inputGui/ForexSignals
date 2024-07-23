import requests
from config import API_URL, HEADERS, OANDA_ACCOUNT_ID

def test_auth():
    endpoint = f"{API_URL}/v3/accounts/{OANDA_ACCOUNT_ID}"
    
    print(f"Testing endpoint: {endpoint}")
    print(f"Headers: {HEADERS}")
    
    try:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        account_data = response.json()
        print("Authentication successful!")
        print(f"Account ID: {account_data['account']['id']}")
        print(f"Account Name: {account_data['account']['alias']}")
        print(f"Account Currency: {account_data['account']['currency']}")
        print(f"Account Balance: {account_data['account']['balance']}")
    except requests.exceptions.RequestException as e:
        print(f"Authentication failed: {e}")
        if response.status_code == 401:
            print("Please check your API key and account ID.")
        print(f"Response content: {response.text}")

if __name__ == "__main__":
    test_auth()