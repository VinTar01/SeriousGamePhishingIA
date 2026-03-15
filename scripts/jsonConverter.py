import pandas as pd
import json
from tqdm import tqdm

#converte in dataset in formato json contenente anche istruzioni per il finetuning
# ======================
# CONFIG
# ======================
INPUT_CSV = "D:/SeriousGamePhishing2/dataset/raw/dataset_finale_ready_v3.csv"
OUTPUT_JSONL = "D:/SeriousGamePhishing2/dataset/processed/dataset_instruction.jsonl"

# ======================
# LOAD
# ======================
df = pd.read_csv(INPUT_CSV)

required_cols = {"subject", "body", "label"}
assert required_cols.issubset(df.columns), "Colonne mancanti nel CSV!"

# ======================
# CONVERSION
# ======================
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row["label"]

        if label == 0:
            instruction = "Write a LEGIT corporate email."
        else:
            instruction = "Write a PHISHING corporate email."

        assistant_output = (
            f"Subject: {row['subject']}\n"
            f"Body:\n{row['body']}"
        )

        example = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": assistant_output}
            ]
        }

        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("Conversione in JSon completata")
print(f"File creato: {OUTPUT_JSONL}")
print(f"Esempi totali: {len(df)}")
