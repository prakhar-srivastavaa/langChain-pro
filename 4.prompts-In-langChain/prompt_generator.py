from langchain_core.prompts import PromptTemplate

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
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

template.save("template.json")