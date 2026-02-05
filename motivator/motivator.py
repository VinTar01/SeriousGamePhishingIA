import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class MotivatorAI:
    def __init__(self, model_path):
        """
        Carica il modello TinyLlama usando i file salvati sul disco.
        """
        print(f">>> [MOTIVATOR] Inizializzazione da: {model_path}")

        # 1. Configurazione Modello Base (Quello su cui hai fatto il fine-tuning)
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # 2. Carica Tokenizer
        # Prima prova a caricarlo dalla tua cartella locale
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            # Se non c'è, lo scarica da internet (è lo stesso)
            print(">>> [MOTIVATOR] Tokenizer locale non trovato, scarico il base...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # 3. Carica il Modello Base
        # Usiamo float16 per compatibilità con la tua GPU Windows
        print(">>> [MOTIVATOR] Caricamento Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 4. Applica il tuo Fine-Tuning (LoRA)
        print(">>> [MOTIVATOR] Applicazione Adattatori LoRA...")
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()  # Modalità inferenza (non training)

        print(">>> [MOTIVATOR] Pronto e operativo!")

    def generate_explanation(self, email_body, is_phishing):
        # Prompt ESATTO usato durante il training (Alpaca Style)
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Cybersecurity Instructor. Analyze the following email and explain briefly why it is PHISHING or LEGIT.

### Input:
{}

### Response:
"""
        # Tagliamo l'email se è troppo lunga per evitare errori di memoria
        prompt = alpaca_prompt.format(email_body[:800])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Risposta breve
                temperature=0.6,  # Creatività bilanciata
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Pulizia: prendiamo solo la parte dopo "### Response:"
        if "### Response:" in decoded:
            response = decoded.split("### Response:")[-1].strip()
        else:
            response = decoded

        return response