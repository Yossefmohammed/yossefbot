# test_hf.py
from huggingface_hub import InferenceClient
import os

# Get token from environment variable (safer than hardcoding)
token = input("Enter your Hugging Face token (starts with hf_): ").strip()

print(f"\n🔍 Testing token: {token[:10]}...")

# Test with public model first (no access required)
try:
    print("\n1️⃣ Testing with public model (Zephyr)...")
    client = InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        token=token,
        timeout=30
    )
    
    response = client.text_generation(
        "Say 'hello' in one word",
        max_new_tokens=10,
        temperature=0.3
    )
    print(f"✅ Success! Response: {response}")
    
except Exception as e:
    print(f"❌ Public model failed: {str(e)}")

# Test with Llama 2 (requires access)
print("\n2️⃣ Testing with Llama 2...")
try:
    client = InferenceClient(
        model="meta-llama/Llama-2-7b-chat-hf",
        token=token,
        timeout=30
    )
    
    response = client.text_generation(
        "Say 'hello' in one word",
        max_new_tokens=10,
        temperature=0.3
    )
    print(f"✅ Success! Response: {response}")
    
except Exception as e:
    print(f"❌ Llama 2 failed: {str(e)}")

# Test token validity
print("\n3️⃣ Testing token with API...")
import requests
headers = {"Authorization": f"Bearer {token}"}

response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
if response.status_code == 200:
    user = response.json()
    print(f"✅ Token valid! Logged in as: {user.get('name', 'Unknown')}")
    print(f"   Email: {user.get('email', 'N/A')}")
    print(f"   Organizations: {[org['name'] for org in user.get('orgs', [])]}")
else:
    print(f"❌ Token invalid! Status: {response.status_code}")
    print(f"   Response: {response.text}")