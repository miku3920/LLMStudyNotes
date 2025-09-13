import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print("OpenAI API Key loaded successfully.")

client = OpenAI(api_key=openai_api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一個專業的程式設計師，精通大型語言模型的應用。"},
            {"role": "user", "content": "你好"}
        ],
        max_tokens=10
    )

    print(f"response：{response}")
    print(f"AI回應：{response.choices[0].message.content}")
except RateLimitError as e:
    print("超出配額")
    print(e)
