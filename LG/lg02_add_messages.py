from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyDate(TypedDict):
    message: Annotated[list, add_messages] #Reducer


from langchain_core.messages import AIMessage, HumanMessage
    
msg1 = [HumanMessage(content="Hello, how are you?", id="1")]
msg2 = [AIMessage(content="Hello, how are you?", id="2")]

result = add_messages(msg1, msg2)   
print(result)  # Output: [HumanMessage(content='Hello, how are you?', id='1'), HumanMessage(content='Hello, how are you?', id='2')]
###############################################
msg3 = [HumanMessage(content="Hello, how are you?", id="3")]
msg4 = [AIMessage(content="Hello, how are you?", id="4")]

result2 = add_messages(msg3, msg4)   
print(result2)  # Output: [HumanMessage(content='Hello, how are you?', id='3'), AIMessage(content='Hello, how are you?', id='4')]
