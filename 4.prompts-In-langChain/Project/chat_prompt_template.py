from langchain_core.prompts import ChatPromptTemplate

chat_template= ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert assistant."),
    ('human', "Explain in simple termas what is {topic}")
])

prompt = chat_template.invoke({"domain":"programming", "topic":"langchain"})

print(prompt)