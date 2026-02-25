from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch
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

detailedPrompt = PromptTemplate(
    template=" Write a detailed report on {topic}",
    input_variables=["topic"]
)

SummarizerPrompt= PromptTemplate(
    template="summarize the following test \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

reportGenerationChain= RunnableSequence(detailedPrompt,model,parser)

branch_chain=RunnableBranch(
    (lambda x : len(x.split()) > 300, RunnableSequence(SummarizerPrompt,model,parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(reportGenerationChain,branch_chain)

result= final_chain.invoke("topic":"russia vs ukraine")

print(result)