import os
import torch
import matplotlib.pyplot as plt
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =============================================================================
# 1. CONFIGURAZIONE SPECIFICA MOTIVATOR
# =============================================================================

HF_TOKEN = "hf...."

# Usiamo TinyLlama: veloce, leggero e ottimo per task semplici come questo
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# PERCORSI (Assicurati che il file JSON sia qui)
# Se il file è nella cartella "motivator", aggiusta il path
JSON_PATH = "C:/Users/daisl/Desktop/SeriousGamePhishing2/motivator/dataset_with_expl.json"
OUTPUT_DIR = "models/tinyllama_motivator_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. FORMATTAZIONE DATASET (EMAIL -> SPIEGAZIONE)
# =============================================================================

def format_motivator_prompt(example):
    """
    Formatta il dataset per insegnare al modello a fare da istruttore.
    Input nel JSON: 'body' (testo email), 'explanation' (risposta desiderata)
    """
    try:
        email_body = example['body']
        explanation = example['explanation']

        # Template "Alpaca-style" semplificato per TinyLlama
        prompt = (
            f"Below is an email. You are a Cybersecurity Instructor. "
            f"Analyze it and explain briefly why it is Phishing or Legit.\n\n"
            f"### Email:\n{email_body}\n\n"
            f"### Explanation:\n{explanation}"
            f"</s>"  # EOS token importante
        )
        return {"text": prompt}
    except Exception as e:
        return {"text": ""}


def get_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir): return None
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]


# =============================================================================
# 3. GRAFICO LOSS
# =============================================================================

def genera_grafico_loss(log_history, output_dir):
    losses = []
    steps = []
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            losses.append(entry['loss'])
            steps.append(entry['step'])

    if not losses: return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', color='#007ACC', linewidth=2)
    plt.title("Training Loss - Motivator AI")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "motivator_loss.png"))
    plt.close()


# =============================================================================
# 4. MAIN TRAINING LOOP
# =============================================================================

def main():
    print("\n=== AVVIO FINE-TUNING MOTIVATOR (TinyLlama) ===\n")

    # 1. CARICAMENTO DATASET
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Dataset non trovato: {JSON_PATH}. Controlla il percorso!")

    print(f">>> Caricamento dataset: {JSON_PATH}")
    dataset = load_dataset("json", data_files=JSON_PATH, split="train")
    print(f"Esempi caricati: {len(dataset)}")

    # 2. FORMATTAZIONE
    print(">>> Formattazione prompt...")
    dataset = dataset.map(format_motivator_prompt)
    print(f"Esempio Prompt:\n{dataset[0]['text'][:300]}...\n")

    # 3. MODELLO E TOKENIZER
    print(f">>> Caricamento Modello: {MODEL_NAME}")
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
        # Target modules specifici per Llama/TinyLlama
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. TRAINING ARGUMENTS
    # Check recovery
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        packing=False,

        # IPERPARAMETRI PER DATASET PICCOLO (60 esempi)
        num_train_epochs=15,  # Più epoche perché i dati sono pochi
        per_device_train_batch_size=2,  # Basso per stabilità
        gradient_accumulation_steps=4,
        learning_rate=2e-4,

        fp16=False,  # Su Windows con GPU NVIDIA recenti, True è meglio. Se dà errore metti False.
        bf16=False,

        logging_steps=5,
        save_strategy="epoch",  # Salva a ogni epoca
        save_total_limit=2,
        warmup_ratio=0.03,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,


    )

    # 6. AVVIO
    print("\nINIZIO TRAINING...\n")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 7. SALVATAGGIO FINALE
    print("\nSalvataggio modello finale...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 8. GRAFICO
    if hasattr(trainer.state, "log_history"):
        genera_grafico_loss(trainer.state.log_history, OUTPUT_DIR)

    print(f"\nMOTIVATOR ADDESTESTRATO! Cartella: {OUTPUT_DIR}")
    print("Ora puoi usare questo modello nello script 'motivator.py'")


if __name__ == "__main__":
    main()