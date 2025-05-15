import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load environment variables
load_dotenv()

class RAGAnswerGenerator:
    def __init__(self):
        """Initialize the RAG answer generator with Hugging Face Gemma-3-4B-IT (8-bit)"""
        model_id = os.getenv("HF_MODEL_ID", "google/gemma-3-4b-it")


        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None
        )

        # Load tokenizer and quantized model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_config,
            device_map="auto"
        )

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and retrieved chunks"""
        context = "\n\n".join([chunk["chunk"] for chunk in retrieved_chunks])
        prompt = self._create_hungarian_prompt(query, context)
        return self._call_llm_local(prompt)

    def _create_hungarian_prompt(self, query: str, context: str) -> str:
        """Create a prompt in Hungarian"""
        return f"""
<start_of_turn>user
Kontextus:
{context}

A fenti információk alapján válaszold meg a következő kérdést magyarul.
Ha a válasz nem található a megadott kontextusban, írd azt, hogy "Nincs elegendő információ a válaszhoz".
Ne találj ki információt, csak a megadott kontextusra hagyatkozz.

Kérdés: {query}
<end_of_turn>
<start_of_turn>model
"""

    def _call_llm_local(self, prompt: str) -> str:
        """Run inference with the local Gemma model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<start_of_turn>model")[-1].strip()

