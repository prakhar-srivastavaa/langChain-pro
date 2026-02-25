from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
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

prompt = PromptTemplate(
    template=" Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

Joke_generatorChain = RunnableSequence(prompt, model, parser)

# def word_count(text):
#     return len(text.split())

parallel_chain= RunnableParallel({
    "joke": RunnablePassthrough(),
    # "word_count": RunnableLambda(word_count) ---#OR#---
    "word_count": RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(Joke_generatorChain,parallel_chain)

result= final_chain.invoke({"topic":"AI"})

finnal_result='''{}\n word count - {}'''.format(result["joke"], result["word_count"])
print(finnal_result)