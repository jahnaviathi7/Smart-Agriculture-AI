import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY not found. Check your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

MODEL = "meta/llama-3.1-8b-instruct"


def ask_agriculture_ai(question):

    system_prompt = """
You are Smart Agriculture AI, an agricultural assistant.

Help farmers with:
- Crop selection
- Crop diseases
- Irrigation
- Fertilizers
- Soil management
- Weather
- Pest management
- Crop care
- Sustainable farming

Give practical, simple and clear answers.

When appropriate:
1. Explain the possible reason.
2. Give practical steps.
3. Mention precautions.
4. Recommend consulting a qualified agricultural expert
   when professional diagnosis is required.

Do not claim certainty when the information is insufficient.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    print("\n======================================")
    print("🤖 SMART AGRICULTURE AI CHATBOT")
    print("======================================")

    question = input(
        "\nAsk your agriculture question: "
    )

    print("\n⏳ Generating answer...\n")

    answer = ask_agriculture_ai(question)

    print("🌾 AI RESPONSE")
    print("--------------------------------------")
    print(answer)