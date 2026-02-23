from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
apikey=os.getenv("OPENROUTER_API_KEY")

prompt1= PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Generate a 5 pointer summary of the following report: {report}",
    input_variables=["report"]
)

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=apikey,
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.5,
    max_tokens=512
)

parser= StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic": "Agentic Ai changes of 2035"})

print(response)

chain.get_graph().print_ascii()