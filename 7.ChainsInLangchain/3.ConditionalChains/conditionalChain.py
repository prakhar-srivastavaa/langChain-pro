from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel , RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

apikey=os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=apikey,
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.5,
    max_tokens=512
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Give the sentiment of the feedback")

parser2= PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=" classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)
positiveFeedbackPropmt = PromptTemplate(
    template="write a approprate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)
negativeFeedbackPrompt = PromptTemplate(
    template="write a approprate response for this negative feedback \n {feedback}",
    input_variables=["feedback"] 
)
#chain
classifier_chain = prompt1 | model | parser2
Branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',positiveFeedbackPropmt | model | parser ),
    (lambda x:x.sentiment=='negative', negativeFeedbackPrompt | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | Branch_chain

result= chain.invoke({"feedback":"This is a good laptop"})

print(result)

chain.get_graph().print_ascii()
