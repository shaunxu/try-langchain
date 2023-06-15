from typing import Optional, List, Any
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer

class ChatGLM(LLM):

    history = []

    tokenizer: Any = None
    model: Any = None

    def __init__(self):
        super().__init__()
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"
    
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=500,
            temperature=0.1
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response
    
    def load_model(self,
                   model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
