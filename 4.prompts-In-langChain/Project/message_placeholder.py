from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_template= ChatPromptTemplate([
    ('system', 'you are a helpful coustomer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

#load chat history
chat_history = []
print("1")
with open('z:/langChain/4.prompts-In-langChain/Project/chat_history.txt') as f:
    chat_history.extend(f.readlines())
    print("3")
print(chat_history)

#create prompt
final_prompt= chat_template.invoke({'chat_history':chat_history, 'query':"where is my refund"})
print(final_prompt)