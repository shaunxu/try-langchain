from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from llms.chatlgm import ChatGLM
import langchain
from langchain.prompts import PromptTemplate

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

请用中文回答，答案的内容不超过200个字。
    """,
    input_variables=["context", "question"]
)

llm = ChatGLM()
llm.load_model(llm_model_name_or_path)

embedding = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
db = Chroma(persist_directory=vectorstore_persist_directory, embedding_function=embedding)

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={
        "prompt": propmt_template
    }
)

print("Ready")
query = "如何计算用户故事的故事点？"
response = chain.run(query)
print(response)


# docs = db.similarity_search(query)
# print(docs[0].metadata)

