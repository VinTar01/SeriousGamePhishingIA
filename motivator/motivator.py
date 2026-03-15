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
        # Usiamo float16 per compatibilità con la tua GPU Windows e risparmiare VRAM
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
        """
        Genera una spiegazione didattica guidata dalla verità (Ground Truth).
        Allineata al formato del dataset v2 (Italiano).
        """

        # 1. Costruiamo un'istruzione specifica in base alla verità nota.
        # Aggiungiamo "Rispondi in italiano" per bloccare l'uso dell'inglese.
        if is_phishing:
            instruction = (
                "Below is an email. You are a Cybersecurity Instructor. This email is a PHISHING ATTACK. "
                "Analyze it and explain briefly why it is suspicious. Rispondi in italiano."
            )
        else:
            instruction = (
                "Below is an email. You are a Cybersecurity Instructor. This email is SAFE and LEGITIMATE. "
                "Analyze it and explain briefly why it looks professional. Rispondi in italiano."
            )

        # 2. TEMPLATE CORRETTO PER IL TUO FINE-TUNING
        # Usiamo esattamente "### Email:" e "### Explanation:" come nel file JSON di addestramento
        prompt = (
            f"{instruction}\n\n"
            f"### Email:\n{email_body[:800]}\n\n"
            f"### Explanation:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 3. Generazione
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,  # Abbastanza per la spiegazione
                temperature=0.3,  # <--- ABBASSATA a 0.3 (Meno creativa, usa ciò che ha studiato)
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4. Pulizia Output basata sul nuovo template
        if "### Explanation:" in decoded:
            response = decoded.split("### Explanation:")[-1].strip()
        else:
            response = decoded.strip()

        return response