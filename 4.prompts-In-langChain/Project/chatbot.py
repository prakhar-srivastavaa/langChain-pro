from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

print("Model loaded successfully!")
print("Type 'exit' or 'quit' to end the conversation.\n")

chat_history=[
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)
