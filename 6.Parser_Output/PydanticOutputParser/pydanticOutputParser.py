from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.5,
    max_tokens=512,
)

class Movie(BaseModel):
    name: str = Field(description="The movie's name")
    year: int = Field(gt=1980, description="The movie's release year, must be greater than 1980")
    rating: int = Field(description="The movie's rating")

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = PromptTemplate.from_template(
    template="Generate the movie name, release year and rating for a movie from {place}.\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# print(prompt.format(place="Indian", format_instructions=parser.get_format_instructions()))

chain = prompt | model | parser

response = chain.invoke({"place": "Indian"})

print(response)