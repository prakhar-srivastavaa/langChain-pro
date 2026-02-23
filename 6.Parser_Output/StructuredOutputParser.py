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

schema=[
    ResponseSchema(name="fact_1", description="A fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="A fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="A fact 3 about the topic"),
    ResponseSchema(name="fact_4", description="A fact 4 about the topic"),
    ResponseSchema(name="fact_5", description="A fact 5 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate.from_template(
    "Provide 5 interesting facts about {topic} in a structured format.\n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser 

result= chain.invoke({"topic": "black hole"})

print(result)