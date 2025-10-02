import os
from dotenv import load_dotenv
import requests

load_dotenv()

def test_huggingface_api():
    # Get token from environment
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ No Hugging Face token found in .env file")
        return False
    
    print(f"✅ Token found: {hf_token[:10]}...")
    
    # Test API call
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    test_text = "Apple stock surges after strong quarterly earnings report"
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": test_text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Result: {result}")
            return True
        elif response.status_code == 503:
            print("⏳ Model is loading, try again in a few minutes")
            return False
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_huggingface_api()