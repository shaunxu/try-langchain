from typing import Optional, List, Any
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AquilaChat(LLM):

    history = []

    tokenizer: Any = None
    model: Any = None

    def __init__(self):
        super().__init__()
    
    @property
    def _llm_type(self) -> str:
        return "AquilaChat"
    
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            ret = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=200,
                use_cache=True
            )
            output_ids = ret[0].detach().cpu().numpy().tolist()
            if 100007 in output_ids:
                output_ids = output_ids[:output_ids.index(100007)]
            else:
                output_ids = output_ids[:output_ids.index(0)]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return response
    
    def load_model(self,
                   model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = self.model.eval()
