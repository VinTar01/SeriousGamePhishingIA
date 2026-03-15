
#Fine tuning sul dataset convertito in formato json

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


#1. CONFIGURAZIONE INIZIALE

HF_TOKEN = "hf...."
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Percorsi Windows
JSONL_PATH = "D:/SeriousGamePhishing2/dataset/processed/dataset_instruction.jsonl"
OUTPUT_DIR = "D:/SeriousGamePhishing2/models/mistral_phishing_v3_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)



# 2. FUNZIONI DI SUPPORTO (RECOVERY & FORMATTAZIONE)

def get_last_checkpoint(output_dir):
    """Cerca l'ultimo checkpoint salvato per riprendere il training."""
    if not os.path.isdir(output_dir):
        return None
    # Cerca cartelle che iniziano con 'checkpoint-'
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # Ordina per numero finale
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]


def format_mistral_chat(example):
    """
    Converte la struttura JSONL 'messages' in stringa piatta per Mistral.
    Input:  [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    Output: <s>[INST] ... [/INST] ...</s>
    """
    try:
        messages = example['messages']
        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        # Formato Mistral v3
        text = f"<s>[INST] {user_content} [/INST] {assistant_content}</s>"
        return {"text": text}
    except Exception as e:
        return {"text": ""}



# 3. GRAFICO LOSS


def genera_grafico_loss(log_history, output_dir):
    """Genera e salva il grafico della loss."""
    losses = []
    steps = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            losses.append(entry['loss'])
            steps.append(entry['step'])

    if not losses:
        print("Nessun dato di loss trovato per il grafico.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', color='#FF5733', linewidth=2)
    plt.title("Andamento Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    path = os.path.join(output_dir, "training_loss_plot.png")
    plt.savefig(path)
    plt.close()
    print(f"Grafico loss salvato in: {path}")



# 4. MAIN


def main():
    print("\n=== AVVIO FINE-TUNING MISTRAL (JSONL MODE) ===\n")

    # 1. CARICAMENTO DATASET (JSONL)

    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"Dataset non trovato: {JSONL_PATH}")

    print(f">>> Caricamento da: {JSONL_PATH}")
    # Usiamo load_dataset di HuggingFace che è ottimizzato
    dataset = load_dataset("json", data_files=JSONL_PATH, split="train")

    print(f"Campioni totali caricati: {len(dataset)}")


    # 2. FORMATTAZIONE (MESSAGES -> TEXT)

    print("\n>>> Formattazione prompt (Chat Template)...")
    # Applichiamo la funzione map per creare la colonna 'text'
    dataset = dataset.map(format_mistral_chat)

    # Esempio di controllo
    print(f"Esempio formattato:\n{dataset[0]['text'][:100]}...")


    # 3. MODELLO + TOKENIZER (QLoRA 4-bit)

    print("\n>>> Caricamento modello base...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="eager"  # Stabilita' su Windows
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # 4. LoRA CONFIG


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. TRAINING CONFIG (SFTConfig)

    # Check Recovery
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f"\n♻️  Trovato checkpoint: {last_checkpoint}. Il training riprenderà da qui!")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Parametri Dataset/Trainer
        dataset_text_field="text",
       # max_seq_length=1024,
        packing=False,

        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2.5e-4,

        # Precisione (IMPORTANTE: fp16=True per evitare OOM su 4-bit loading)
        fp16=False,
        bf16=False,

        logging_steps=10,
        save_strategy="steps",
        save_steps=100,  # Salva ogni 100 step per recovery
        save_total_limit=2,  # Tiene solo gli ultimi 2 checkpoint

        warmup_ratio=0.03,
        max_grad_norm=0.3,
        group_by_length=False,  # False per stabilità
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,  # processing_class sostituisce 'tokenizer' nelle nuove versioni
        args=training_args,
        peft_config=peft_config,
    )


    # 6. TRAINING

    print("\n INIZIO TRAINING...\n")
    # Passiamo il checkpoint se esiste
    trainer.train(resume_from_checkpoint=last_checkpoint)


    # 7. SALVATAGGIO

    print("\n Salvataggio modello finale...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


    # 8. REPORT GRAFICO

    print("\n Generazione grafico loss...")
    if hasattr(trainer.state, "log_history"):
        genera_grafico_loss(trainer.state.log_history, OUTPUT_DIR)

    print(f"\n FINE PROCEDURA. Modello salvato in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()