from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set huggingface cache home
os.environ["HF_HOME"] = "Z:/huggingface_cache"

model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

print("Loading model and tokenizer...")

# Load the model using HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

# Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain."),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)