## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
Modern large language models (LLMs) are powerful at generating human-like responses, but they often lack grounding in specific knowledge sources. This leads to issues such as hallucination, where the model generates information that is not factually correct. To address this, there is a need for a system that can combine retrieval of relevant documents with LLM reasoning, ensuring that answers are accurate and context-driven.

### DESIGN STEPS:

#### STEP 1:
Embed and store documents in a vector database.
#### STEP 2:
Retrieve relevant documents and build a prompt with context + question.
#### STEP 3:
Run the LCEL chain (Retriever → Prompt → Model → Output) to generate the answer.
### PROGRAM:
### Simple Chain:
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

#!pip install pydantic==1.10.8

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()
chain = prompt | model | output_parser
chain.invoke({"topic": "puppies"})
```

### More complex chain:
```
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
vectorstore = DocArrayInMemorySearch.from_texts(
    ["she went to school everyday", "The new movie is coming soon"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("she went to school everyday")
retriever.get_relevant_documents("The new movie is coming soon")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.schema.runnable import RunnableMap
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser
chain.invoke({"question": "where did she go everyday?"})
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})
inputs.invoke({"question": "where did she go everyday?"})
```

### OUTPUT:
<img width="831" height="456" alt="image" src="https://github.com/user-attachments/assets/9d7760a7-38ad-4adc-9b11-03d2f0d5f0ce" />

<img width="1016" height="846" alt="Screenshot 2025-09-19 111507" src="https://github.com/user-attachments/assets/8b60f14a-7fce-441e-9017-faaa44fe76ad" />

### RESULT:
Thus, The implementation of a LangChain Expression Language (LCEL) is successfully executed.
