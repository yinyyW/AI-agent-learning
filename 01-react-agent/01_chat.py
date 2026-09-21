from openai import OpenAI
from dotenv import load_dotenv
import os

# get api key from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# create LLM client
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# chat with LLM
response = client.responses.create(
    model="你的模型",
    input="What is the capital of France?"
)

print(response.output_text)