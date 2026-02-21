import random
import re
from datasets import load_dataset, concatenate_datasets, ClassLabel

# =====================================================
# 1. LOAD DATASET
# =====================================================

dataset = load_dataset("medmcqa")

if "validation" in dataset:
    train_data = concatenate_datasets([dataset["train"], dataset["validation"]])
else:
    train_data = dataset["train"]

# =====================================================
# 2. UTILITY FUNCTIONS
# =====================================================

def filter_by_answer(example, keywords):
    options = [
        example.get("opa", ""),
        example.get("opb", ""),
        example.get("opc", ""),
        example.get("opd", "")
    ]
    question = example.get("question", "")
    text_to_check = " ".join(options + [question]).lower()

    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", text_to_check):
            return True
    return False


def clean_question(text):
    text = re.sub(r"What is the most likely diagnosis\??.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Which of the following.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Choose the correct answer.*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_case(example, disease_name):
    cleaned_question = clean_question(example["question"])
    return {
        "symptoms_text": cleaned_question,
        "disease": disease_name,
        "presentation_type": "typical"
    }


def long_vignette(example):
    return len(example["symptoms_text"].split()) > 8


# =====================================================
# 3. FILTER BY DISEASE KEYWORDS
# =====================================================

mi_dataset = train_data.filter(
    lambda x: filter_by_answer(x, [
        "myocardial infarction",
        "acute myocardial infarction",
        "stemi",
        "nstemi",
        "acute coronary syndrome",
        "heart attack"
    ])
)

stroke_dataset = train_data.filter(
    lambda x: filter_by_answer(x, [
        "stroke",
        "ischemic stroke",
        "cerebral infarction",
        "intracerebral hemorrhage",
        "cva"
    ])
)

tb_dataset = train_data.filter(
    lambda x: filter_by_answer(x, [
        "tuberculosis",
        "pulmonary tuberculosis",
        "tuberculous"
    ])
)

diabetes_dataset = train_data.filter(
    lambda x: filter_by_answer(x, [
        "diabetes mellitus",
        "diabetic ketoacidosis",
        "hypoglycemia",
        "hyperglycemia"
    ])
)

# =====================================================
# 4. EXTRACT & CLEAN
# =====================================================

mi_cases = mi_dataset.map(lambda x: extract_case(x, "MI"), remove_columns=mi_dataset.column_names)
stroke_cases = stroke_dataset.map(lambda x: extract_case(x, "Stroke"), remove_columns=stroke_dataset.column_names)
tb_cases = tb_dataset.map(lambda x: extract_case(x, "TB"), remove_columns=tb_dataset.column_names)
diabetes_cases = diabetes_dataset.map(lambda x: extract_case(x, "Diabetes"), remove_columns=diabetes_dataset.column_names)

# =====================================================
# 5. LENGTH FILTER
# =====================================================

mi_cases = mi_cases.filter(long_vignette)
stroke_cases = stroke_cases.filter(long_vignette)
tb_cases = tb_cases.filter(long_vignette)
diabetes_cases = diabetes_cases.filter(long_vignette)

print("Counts After Filtering:")
print("MI:", len(mi_cases))
print("Stroke:", len(stroke_cases))
print("TB:", len(tb_cases))
print("Diabetes:", len(diabetes_cases))

# =====================================================
# 6. UPSAMPLE TO BALANCE
# =====================================================

def upsample_dataset(ds, target_n, seed=42):
    n = len(ds)
    if n == 0:
        return ds
    if n >= target_n:
        return ds.shuffle(seed=seed).select(range(target_n))

    parts = []
    repeats = target_n // n
    rem = target_n % n
    for _ in range(repeats):
        parts.append(ds)
    if rem:
        parts.append(ds.shuffle(seed=seed).select(range(rem)))

    return concatenate_datasets(parts)


counts = [len(mi_cases), len(stroke_cases), len(tb_cases), len(diabetes_cases)]
target = min(max(counts), 200)

mi_cases = upsample_dataset(mi_cases, target)
stroke_cases = upsample_dataset(stroke_cases, target)
tb_cases = upsample_dataset(tb_cases, target)
diabetes_cases = upsample_dataset(diabetes_cases, target)

print("Counts After Upsampling:")
print("MI:", len(mi_cases))
print("Stroke:", len(stroke_cases))
print("TB:", len(tb_cases))
print("Diabetes:", len(diabetes_cases))

# =====================================================
# 7. BUILD TYPICAL DATASET
# =====================================================

typical_dataset = concatenate_datasets([
    mi_cases,
    stroke_cases,
    tb_cases,
    diabetes_cases
])

print("Typical Dataset Size:", len(typical_dataset))

# =====================================================
# 8. GENERATE ATYPICAL DATASET
# =====================================================

canonical_symptoms = {
    "MI": ["chest pain", "substernal", "radiating", "left arm", "crushing", "diaphoresis"],
    "Stroke": ["hemiplegia", "facial droop", "slurred speech", "weakness"],
    "TB": ["chronic cough", "hemoptysis", "night sweats", "weight loss"],
    "Diabetes": ["polyuria", "polydipsia", "fruity breath", "sweating"]
}

def remove_canonical(text, disease):
    modified = text
    for symptom in canonical_symptoms[disease]:
        modified = re.sub(r"\b" + re.escape(symptom) + r"\b", "", modified, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", modified).strip()

def generate_atypical(example):
    return {
        "symptoms_text": remove_canonical(example["symptoms_text"], example["disease"]),
        "disease": example["disease"],
        "presentation_type": "atypical"
    }

atypical_dataset = typical_dataset.map(generate_atypical)

# =====================================================
# 9. FINAL DATASET
# =====================================================

final_dataset = concatenate_datasets([typical_dataset, atypical_dataset])

print("Final Dataset Size:", len(final_dataset))

# Add numeric ClassLabel column for stratified splitting
label_names = ["MI", "Stroke", "TB", "Diabetes"]
label_map = {n: i for i, n in enumerate(label_names)}

final_dataset = final_dataset.map(lambda x: {"disease_label": label_map.get(x.get("disease", ""), -1)})
final_dataset = final_dataset.cast_column("disease_label", ClassLabel(names=label_names))

# =====================================================
# 10. FORMAT FOR TRAINING (4-CLASS CLASSIFICATION)
# =====================================================

def format_for_training(example):
    prompt = f"""Patient Case:
{example['symptoms_text']}

Which disease does this patient most likely have?
Options: MI, Stroke, TB, Diabetes"""

    return {
        "input_text": prompt,
        "label": example["disease"],
        "presentation_type": example["presentation_type"]
    }

split = final_dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="disease_label"
)

train_ds = split["train"].map(format_for_training, remove_columns=split["train"].column_names)
test_ds = split["test"].map(format_for_training, remove_columns=split["test"].column_names)

print("Train size:", len(train_ds))
print("Test size:", len(test_ds))

print(train_ds[0])
train_ds.save_to_disk("data/processed/train")
test_ds.save_to_disk("data/processed/test")