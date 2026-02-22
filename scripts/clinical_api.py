from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from clinical_structuring_engine import run_clinical_pipeline, load_medgemma_lora

app = FastAPI()

class DiagnoseRequest(BaseModel):
    raw_input_text: str

# Load model and tokenizer once
TOKENIZER, MODEL = load_medgemma_lora()

@app.post("/diagnose")
def diagnose(request: DiagnoseRequest) -> Dict[str, Any]:
    result = run_clinical_pipeline(request.raw_input_text, TOKENIZER, MODEL)
    return result

if __name__ == "__main__":
    uvicorn.run("scripts.clinical_api:app", host="0.0.0.0", port=8000, reload=True)
