from typing import Dict, List, Tuple, Any
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Case Structuring Module ---
def rule_based_structuring(raw_text: str) -> Dict[str, Any]:
    # Simple regex/keyword extraction
    age = re.search(r'age\s*:?\s*(\d+)', raw_text, re.I)
    gender = re.search(r'(male|female|m|f)', raw_text, re.I)
    symptoms = re.findall(r'symptom[s]?\s*:?\s*([\w, ]+)', raw_text, re.I)
    duration = re.search(r'duration\s*:?\s*([\w ]+)', raw_text, re.I)
    labs = re.findall(r'lab[s]?\s*:?\s*([\w, ]+)', raw_text, re.I)
    history = re.search(r'history\s*:?\s*([\w, ]+)', raw_text, re.I)
    return {
        'age': int(age.group(1)) if age else None,
        'gender': gender.group(1).lower() if gender else None,
        'symptoms': symptoms[0].split(',') if symptoms else [],
        'duration': duration.group(1) if duration else None,
        'labs': labs[0].split(',') if labs else [],
        'history': history.group(1) if history else None,
    }

def llm_structuring(raw_text: str, tokenizer, model) -> Dict[str, Any]:
    # Optionally use LLM for structuring
    prompt = f"Extract age, gender, symptoms, duration, labs, history from: {raw_text}\nOutput JSON."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return eval(result)
    except Exception:
        return {}

def structure_case(raw_text: str, tokenizer=None, model=None) -> Dict[str, Any]:
    structured = rule_based_structuring(raw_text)
    if tokenizer and model:
        llm_structured = llm_structuring(raw_text, tokenizer, model)
        structured.update({k: v for k, v in llm_structured.items() if v})
    return structured

# --- Atypical Sensitivity Analyzer ---
def remove_canonical_symptom(structured_case: Dict[str, Any]) -> Dict[str, Any]:
    # Remove first symptom for sensitivity analysis
    new_case = structured_case.copy()
    if new_case['symptoms']:
        new_case['symptoms'] = new_case['symptoms'][1:]
    return new_case

def compute_robustness(original_pred: str, altered_pred: str) -> float:
    return float(original_pred == altered_pred)

# --- Differential Diagnosis Generator ---
def get_top_differentials(prompt: str, tokenizer, model) -> List[Tuple[str, float]]:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, num_return_sequences=3)
    diagnoses = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    # Dummy confidence scores
    return [(d, 0.8 - 0.1*i) for i, d in enumerate(diagnoses)]

# --- Suggested Next Steps Module ---
def suggest_next_steps(disease: str) -> Dict[str, List[str]]:
    # Simple mapping
    suggestions = {
        'diabetes': {
            'tests': ['HbA1c', 'Fasting glucose'],
            'red_flags': ['DKA', 'hypoglycemia']
        },
        'pneumonia': {
            'tests': ['Chest X-ray', 'CBC'],
            'red_flags': ['Sepsis', 'respiratory failure']
        },
    }
    return suggestions.get(disease.lower(), {'tests': [], 'red_flags': []})

# --- Clarification Step ---
def generate_clarification(prompt: str, tokenizer, model) -> str:
    clar_prompt = f"If confidence < 0.6, ask a clarifying question for: {prompt}"
    inputs = tokenizer(clar_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
