from transformers import AutoModel, AutoTokenizer
from langchain import HuggingFacePipeline

model_path = "/Users/shaunxu/models/chatglm-6b-int4"

# model = HuggingFacePipeline.from_model_id(model_id=model_path,
#             task="text-generation",
#             model_kwargs={
#                           "torch_dtype" : load_type,
#                           "low_cpu_mem_usage" : True,
#                           "temperature": 0.2,
#                           "max_length": 1000,
#                           "device_map": "auto",
#                           "repetition_penalty":1.1,
#                           "trust_remote_code": True}
#             )

tokenizer = AutoTokenizer.from_pretrained("/Users/shaunxu/models/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("/Users/shaunxu/models/chatglm-6b-int4", trust_remote_code=True, ignore_mismatched_sizes=True).float()
model = model.eval()

prompt = "你好"
print(f"Human: {prompt}")

response, _ = model.chat(tokenizer, prompt, history=[], max_length=500, temperature=0.2)
print(f"AI: {response}")

# from to_upper_llm import ToTupperLLM
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.agents import load_tools, initialize_agent, AgentType
# from langchain import ConversationChain

# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="I'm a human and I said \"{input}\"."
# )

# print(prompt.format(input="hello"))

# llm = ToTupperLLM()

# chain = ConversationChain(llm=llm, verbose=True)
# output = chain.predict(input="hello")
# print(output)
# output = chain.predict(input="hello")
# print(output)

# tools = load_tools(["human", "llm-math"], llm=llm)
# agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("What's my friend Eric's surname?")

# chain = LLMChain(llm=llm, prompt=prompt)
# response = chain.run("hello")
# print(response)

# input = "This is a foobar thing"
# output = llm(input)
# print("🧑‍💼: " + input + "\n" + "🤖: " + output)