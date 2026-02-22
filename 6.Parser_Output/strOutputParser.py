from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model= ChatHuggingFace(llm=llm)

#1st prompt -> detailsed report
template1= PromptTemplate(
    template= "write a detailed report on the following {topic}",
    input_variables=["topic"]
)

#2nd prompt -> summary
template2= PromptTemplate(
    template= "write a 5 line summary of the following {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

res=chain.invoke({"topic": "black hole"})

print(res)