from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from llms.chatlgm import ChatGLM
import langchain
from langchain.prompts import PromptTemplate
import datetime
from typing import Union
from typing import TypeVar

# T = TypeVar("T")

langchain.debug = True

load_dotenv()
llm_model_name_or_path = os.environ.get("LLM_MODEL_NAME_OR_PATH")
embedding_model_name_or_path = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH")
vectorstore_persist_directory = os.environ.get("VECTORSTORE_PERSIST_DIRECTORY")

propmt_template = PromptTemplate(
    template="""请使用如下信息来回答问题。如果你无法从信息中获得答案，请回答“我不知道”，不要自行杜撰答案。

{context}

问题如下。

{question}

请用中文回答，答案的内容不超过100个字。
    """,
    input_variables=["context", "question"]
)

print(f"[{datetime.datetime.now()}] embedding")
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
print(f"[{datetime.datetime.now()}] db")
db = Chroma(persist_directory=vectorstore_persist_directory, embedding_function=embedding)

question = "如何计算故事点？"

print(f"[{datetime.datetime.now()}] db.similarity_search")
docs = db.similarity_search(question, k=4)

context = ""
for doc in docs:
    context += f"{doc.page_content}\n"

print(f"[{datetime.datetime.now()}] llm")
llm = ChatGLM()
llm.load_model(llm_model_name_or_path)

print(f"[{datetime.datetime.now()}] chain")
chain = LLMChain(
    llm=llm,
    prompt=propmt_template
)

print(f"[{datetime.datetime.now()}] predict")
output = chain.predict(context=context, question=question)

print(output)