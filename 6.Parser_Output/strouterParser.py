from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
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

prompt1= template1.invoke({"topic":"The impact of climate change on global agriculture"})
result= model.invoke(prompt1)
prompt2= template2.invoke({"text":result.content})
summary= model.invoke(prompt2) 

print(summary.content)