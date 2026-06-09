from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

client=ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY")
)

conversation = client.chat.completions.create(
    model='gemini-2.5-flash',
    messages=[
        {"role":"system", "content":"You are a helpful assistant"},
        {"role":"user", "content":"What is the capital of France?"}
    ]
)

print(conversation.choices[0].message.content)



client.embeddings.create(
    input="This is a good conversation", model=
)