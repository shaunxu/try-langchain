from dotenv import load_dotenv
import os
import datetime
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from llms.chatglm import ChatGLM
from llms.moss import Moss
from llms.aquilachat import AquilaChat
from llms.chinese_llama_alpaca import ChineseLlamaAlpaca
from langchain import HuggingFacePipeline
import torch

load_dotenv()
llm_model_name_or_path = os.environ.get("LLM_MODEL_NAME_OR_PATH")
embedding_model_name_or_path = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH")
vectorstore_persist_directory = os.environ.get("VECTORSTORE_PERSIST_DIRECTORY")

def print_with_timeframe(message: str) -> None:
    print(f"[{datetime.datetime.now()}] {message}")

def chat(chain: ConversationChain, question: str) -> None:
    response = chain.predict(input=question)
    print_with_timeframe(f"🤖: {response}")

def main():
    prompt = PromptTemplate(
        template="""
你的名字叫“小P”，是PingCode智能助手，正在和人类进行多轮对话。
你要简单扼要的回答人类提出的问题。如果你不知道答案，请说我不知道，不要杜撰。
注意，如果问题或答案涉及暴力、色情、毒品等违法犯罪情形，请拒绝回答。
同时，请控制回答的长度不要超过100个字。
以下是之前问答的信息，在<<<和>>>之间。
<<<
{history}
>>>
以下是人类提出的问题，在连续三个反引号之间。
```
{input}
```
""",
        input_variables=["history", "input"],
    )

    print_with_timeframe(f"Initializing LLM from {llm_model_name_or_path}")
    # llm = ChatGLM()
    # llm = Moss()
    # llm = AquilaChat()
    # llm = ChineseLlamaAlpaca()
    # llm.load_model(llm_model_name_or_path)

    llm = HuggingFacePipeline.from_model_id(model_id=llm_model_name_or_path,
                                            task="text-generation",
                                            model_kwargs={
                                                "torch_dtype" : torch.float16,
                                                "low_cpu_mem_usage" : True,
                                                "temperature": 0.2,
                                                "max_length": 1000,
                                                "device_map": "auto",
                                                "repetition_penalty":1.1}
    )

    print_with_timeframe(f"LLM is ready. LLM={llm._llm_type}")

    print_with_timeframe(f"Initializing chain with prompt")
    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationBufferWindowMemory(k=10),
        verbose=False
    )

    print_with_timeframe(f"Ready")
    chat(chain, "你好")
    while True:
        question = input(f"[{datetime.datetime.now()}] 🧑‍💼: ")
        if question == "exit":
            break
        elif question == "":
            continue
        else:
            chat(chain, question)

if __name__ == "__main__":
    main()
