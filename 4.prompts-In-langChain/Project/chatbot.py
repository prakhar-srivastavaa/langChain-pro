# from langchain_huggingface import ChatHuggingFace
# import os

# # Set huggingface cache home
# os.environ["HF_HOME"] = "Z:/huggingface_cache"

# model= ChatHuggingFace.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation", 
#     pipeline_kwargs=dict(
#         temperature=0.7,    
#         max_new_tokens=100
#     )
# )//
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

# Set huggingface cache home
os.environ["HF_HOME"] = "Z:/huggingface_cache"

model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

print("Loading model and tokenizer...")
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    
    model_name,
    torch_dtype="bfloat16",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model loaded successfully! Device: {model.device}")
print("Type 'exit' or 'quit' to end the conversation.\n")


chat_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Generate response
    output = model.generate(
        input_ids.to(model.device),
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.95
    )
    
    # Decode the response
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract only the assistant's response (after the last [|assistant|] tag)
    assistant_response = full_response.split("[|assistant|]")[-1].strip()
    
    # Add assistant response to chat history
    chat_history.append({"role": "assistant", "content": assistant_response})
    
    print(f"AI: {assistant_response}\n")