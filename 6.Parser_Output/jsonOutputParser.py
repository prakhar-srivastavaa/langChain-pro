from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
    }
)
model = ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template= PromptTemplate(
    template= " Give me 5 facts about {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)
# final_json_result = parser.parse(result.content)
# print(final_json_result)
# print(type(final_json_result))

# using chaain#########
chain= template | model | parser
final_result= chain.invoke({"topic": "black hole"})
print(final_result)