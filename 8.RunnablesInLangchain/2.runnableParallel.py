from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
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

prompt1= PromptTemplate(
    template="Genarate a tweet about {topic}",
    input_variables=["topic"]
)
prompt2= PromptTemplate(
    template="Genarate a linkedin post about {topic}",
    input_variables=["topic"]
)
parser= StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin':RunnableSequence(prompt2, model, parser)
})

result=parallel_chain.invoke({"topic":"AI"})
print(result)
print(result["tweet"])
