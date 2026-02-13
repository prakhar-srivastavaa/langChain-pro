from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace
import os
# Set huggingface cache home
os.environ["HF_HOME"] = "Z:/huggingface_cache"

model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

print("Loading model and tokenizer...")

# Load the model and tokenizer
model = ChatHuggingFace.from_model_id(
    model_id=model_name,
    task="text-generation",
    device_map="auto"
)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain."),
]

result=model.invoke(messages)
messages.append(AIMessage(result.content))

print(messages)