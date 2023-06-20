from typing import Optional, List, Any
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChineseLlamaAlpaca(LLM):

    history = []

    tokenizer: Any = None
    model: Any = None

    def __init__(self):
        super().__init__()
    
    @property
    def _llm_type(self) -> str:
        return "Chinese Llama Alpaca"
    
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].cuda()
        outputs = self.model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
        # response, history = self.model.chat(
        #     self.tokenizer,
        #     prompt,
        #     history=self.history,
        #     max_length=10000,
        #     temperature=0.1
        # )
        # if stop is not None:
        #     response = enforce_stop_tokens(response, stop)
        # self.history = self.history + [[None, response]]
        # return response
    
    def load_model(self,
                   model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().cuda()
        self.model = self.model.eval()
