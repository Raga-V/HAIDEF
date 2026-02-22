import os
import json
from typing import Dict, Any
from dataset_loader import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from clinical_modules import (
    structure_case,
    remove_canonical_symptom,
    compute_robustness,
    get_top_differentials,
    suggest_next_steps,
    generate_clarification
)

# Path setup
data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
model_dir = os.path.join(os.path.dirname(__file__), '../models/lora')

def load_ehr_data(split='test'):
    split_dir = os.path.join(data_dir, split)
    arrow_file = os.path.join(split_dir, 'data-00000-of-00001.arrow')
    info_file = os.path.join(split_dir, 'dataset_info.json')
    state_file = os.path.join(split_dir, 'state.json')
    return load_dataset(arrow_file, info_file, state_file)

def load_medgemma_lora():

    base_model_name = "google/medgemma-4b-it"
    lora_dir = os.path.abspath("../models/lora")  # your saved adapter path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()

    return tokenizer, model

# --- Agentic Workflow Orchestrator ---
def run_clinical_pipeline(raw_input_text: str, tokenizer, model) -> Dict[str, Any]:
    # 1. Structure input
    structured_case = structure_case(raw_input_text, tokenizer, model)
    # 2. Model inference (top-3 differentials)
    prompt = f"Patient: {structured_case}\nDiagnose:"
    top_3 = get_top_differentials(prompt, tokenizer, model)
    predicted_diagnosis, confidence = top_3[0]
    # 3. Robustness (atypical sensitivity)
    altered_case = remove_canonical_symptom(structured_case)
    altered_prompt = f"Patient: {altered_case}\nDiagnose:"
    altered_top_3 = get_top_differentials(altered_prompt, tokenizer, model)
    atypical_score = compute_robustness(predicted_diagnosis, altered_top_3[0][0])
    # 4. Recommendations
    recommendations = suggest_next_steps(predicted_diagnosis)
    # 5. Clarification step
    clarifying_question = None
    if confidence < 0.6:
        clarifying_question = generate_clarification(prompt, tokenizer, model)
    # 6. Output
    return {
        'structured_case': structured_case,
        'predicted_diagnosis': predicted_diagnosis,
        'top_3_differentials': top_3,
        'atypical_score': atypical_score,
        'recommendations': recommendations,
        'clarifying_question': clarifying_question
    }

def clinical_structuring_engine():
    tokenizer, model = load_medgemma_lora()
    ehr_data = load_ehr_data('test')
    results = []
    for record in ehr_data:
        raw_text = record.get('notes', '')
        output = run_clinical_pipeline(raw_text, tokenizer, model)
        results.append(output)
    with open(os.path.join(os.path.dirname(__file__), '../results/clinical_structuring_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Clinical structuring and diagnosis complete.")

if __name__ == '__main__':
    clinical_structuring_engine()
