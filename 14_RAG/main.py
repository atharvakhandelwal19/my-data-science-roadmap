import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from importlib.metadata import version


core_version = version("langchain-core")
graph_version = version("langgraph")

print(core_version, graph_version)
load_dotenv()


# 3. Test the connection
def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke([{"role":"user", "content":"Hello,World"}])
    print(response)

    print('Generated')

if __name__ == "__main__":
    main()