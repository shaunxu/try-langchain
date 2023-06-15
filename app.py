from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
import os
from llms.chatlgm import ChatGLM
from transformers import AutoModel, AutoTokenizer

load_dotenv()

model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH")

def main():
    llm = ChatGLM()
    llm.load_model(model_name_or_path)

    template = """你是一个友好的人工智能机器人，正在和一个人类对话。
你将会尽量详细且正确的回答人类提出的问题。如果你不知道如何回答，请说我不知道。
这是人类提出的问题。
{question}
"""
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    response = chain.predict(question="请介绍一下北京这座城市。")
    print(response)
    

if __name__ == "__main__":
    main()

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
# model = model.eval()

# prompt = "你好"
# print(f"Human: {prompt}")
# response, _ = model.chat(tokenizer, prompt, history=[], max_length=500, temperature=0.1)
# print(f"AI: {response}")
