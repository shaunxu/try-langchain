from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
import os
from llms.chatglm import ChatGLM
import time
from typing import Union
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
llm_model_name_or_path = os.environ.get("LLM_MODEL_NAME_OR_PATH")
embedding_model_name_or_path = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH")
vectorstore_persist_directory = os.environ.get("VECTORSTORE_PERSIST_DIRECTORY")

def ask(chain: Chain, question: str) -> Union[str, float]:
    start = time.time()
    response = chain.predict(input=question)
    # response = chain.run(question)
    end = time.time()
    return [response, end - start]


def print_response(message: str, elasped: float) -> None:
    print(f"🤖: {message}")
    print(f"🕗: {elasped}")

def main():
    llm = ChatGLM()
    llm.load_model(llm_model_name_or_path)

    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
    db = Chroma(persist_directory=vectorstore_persist_directory, embedding_function=embedding)

    template = """你的名字叫“小P”，是PingCode智能助手，正在和人类进行多轮对话。
你要简单扼要的回答人类提出的问题。
如果你不知道答案，请说我不知道，不要杜撰。
注意，如果问题或答案涉及暴力、色情、毒品等违法犯罪情形，请拒绝回答。
同时，请控制回答的长度不要超过200个字。
以下是之前问答的信息。
{history}
以下是人类提出的问题。
{input}
"""
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    # chain = LLMChain(
    #     llm=llm,
    #     prompt=prompt,
    #     verbose=False
    # )

    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        # memory=ConversationBufferMemory(),
        memory=ConversationBufferWindowMemory(k=20),
        # memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=500),
        verbose=False
    )

    # chain = RetrievalQA(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=db.as_retriever()
    # )
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm, 
    #     chain_type="stuff",
    #     retriever=db.as_retriever()
    # )

    response, elasped = ask(chain, "你好")
    print_response(response, elasped)

    while True:
        question = input("🧑‍💼: ")
        if question == "exit":
            print_response("Bye~", 0)
            break
        elif question == "":
            continue
        else:
            response, elasped = ask(chain, question)
            print_response(response, elasped)

            # start = time.time()
            # response = chain.predict(input=question)
            # end = time.time()
            # print_response(response, end - start)

if __name__ == "__main__":
    main()

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
# model = model.eval()

# prompt = "你好"
# print(f"Human: {prompt}")
# response, _ = model.chat(tokenizer, prompt, history=[], max_length=500, temperature=0.1)
# print(f"AI: {response}")
