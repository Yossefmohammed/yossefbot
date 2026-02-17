from groq import Groq
import os

# Hardcode your API key for testing (remove after testing!)
import os
API_KEY = os.getenv("GROQ_API_KEY")  # Replace with your actual key

print("1️⃣ Testing Groq connection...")
try:
    client = Groq(api_key=E)
    
    # Test with a simple completion
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Say 'hello' in one word"}],
        temperature=0.3,
        max_tokens=10
    )
    
    print(f"✅ Success! Response: {completion.choices[0].message.content}")
    print(f"Model used: llama3-8b-8192")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")