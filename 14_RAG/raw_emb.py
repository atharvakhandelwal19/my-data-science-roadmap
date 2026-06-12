from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# conversation = client.chat.completions.create(
#     model="gemini-2.5-flash",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France?"},
#     ]
# )

# print(conversation.choices[0].message.content)


response = client.embeddings.create(
    input="This is a good conversation", 
    model="gemini-embedding-2" 
)


embedding_vector = response.data[0].embedding

print(f"Dimensions: {len(embedding_vector)}")
print('='*100)
print(f"First 5 values: {embedding_vector[:5]}")
