import requests
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGAnswerGenerator:
    def __init__(self):
        """Initialize the RAG answer generator"""
        # Get API key from environment
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            print("Warning: No LLM_API_KEY found in environment variables")
        
        # You can use any LLM API that supports Hungarian
        # This example uses a placeholder for a generic API
        self.api_url = os.getenv("LLM_API_URL", "https://api.your-llm-provider.com/v1/completions")
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and retrieved chunks"""
        # Extract text from chunks
        context_texts = [chunk["chunk"] for chunk in retrieved_chunks]
        context = "\n\n".join(context_texts)
        
        # Create prompt
        prompt = self._create_hungarian_prompt(query, context)
        
        # Call LLM API
        try:
            return self._call_llm_api(prompt)
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "Sajnos nem tudtam választ generálni. Kérlek próbáld újra később."
    
    def _create_hungarian_prompt(self, query: str, context: str) -> str:
        """Create a prompt in Hungarian for the LLM"""
        return f"""
Kontextus:
{context}

A fenti információk alapján válaszold meg a következő kérdést magyarul.
Ha a válasz nem található a megadott kontextusban, írd azt, hogy "Nincs elegendő információ a válaszhoz".
Ne találj ki információt, csak a megadott kontextusra hagyatkozz.

Kérdés: {query}

Válasz:
"""
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call the LLM API with the prompt"""
        # If no API key is provided, return a placeholder response
        if not self.api_key:
            return "(Placeholder válasz - API kulcs hiányzik)"
        
        # Example implementation for a generic API
        try:
            payload = {
                "model": "your-hungarian-model",
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return response.json().get("choices")[0].get("text", "").strip()
            else:
                print(f"API error: {response.status_code}, {response.text}")
                return "API hiba történt. Kérlek próbáld újra később."
                
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return "Hiba történt a válasz generálása közben."
            
    def query_without_context(self, query: str) -> str:
        """Generate an answer without any context - for testing"""
        prompt = f"Válaszolj a következő kérdésre magyarul: {query}"
        return self._call_llm_api(prompt)
