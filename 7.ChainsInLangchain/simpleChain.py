from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.5,
    max_tokens=200,
)

prompt= PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "langchain"})

print(response)

chain.get_graph().print_ascii()