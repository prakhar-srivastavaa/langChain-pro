from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

apikey=os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=apikey,
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.5,
    max_tokens=512
)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain the folloeing joke {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model , parser, prompt2, model , parser)

result = chain.invoke({'topic':"AI"})

print(result)