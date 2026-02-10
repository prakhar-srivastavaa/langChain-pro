from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
os.environ["HF_Home"] = "Z:/huggingface_cache"

llm= HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model= ChatHuggingFace(llm=llm)
result= model.invoke("What is the capital of India?")
print(result.content)

#  output:
# <|user|>
# What is the capital of India?</s>
# <|assistant|>
# India's capital city is Delhi, also known as New Delhi.
