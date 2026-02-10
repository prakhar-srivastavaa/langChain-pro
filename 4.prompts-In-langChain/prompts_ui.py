from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
import os
import torch
from langchain_core.prompts import PromptTemplate

# Set huggingface cache home
os.environ["HF_HOME"] = "Z:/huggingface_cache"


@st.cache_resource
def load_model():
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        device=device,
        pipeline_kwargs=dict(
            temperature=0.7,
            max_new_tokens=100,
            do_sample=True
        )
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

st.header("Research Tool")

# prompt = st.text_area("Enter your research question or topic here:", height=100)

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template
template_str = """
Summarize the research paper "{paper_input}" according to the following requirements:
- **Style**: {style_input}
- **Length**: {length_input}

Guidelines:
1. If the style is technical or mathematical, include relevant equations or technical terminology.
2. If the style is beginner-friendly, use analogies and avoid heavy jargon.
3. If the style is code-oriented, focus on the algorithm and provide pseudocode.
4. If information is not available in the paper, respond with "Insufficient information to answer the question".

Summary:
"""

template = PromptTemplate(
    template=template_str,
    input_variables=["paper_input", "style_input", "length_input"]
    validate_template=True
)

# fill the placeholders 
prompt=template.invoke({
    "paper_input":paper_input,
    "style_input":style_input, 
    "length_input":length_input
    })

if st.button("Summarize"):
    result= model.invoke(prompt)
    st.write(result.content)