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
        """

        # 1. Costruiamo un'istruzione specifica in base alla verità nota.
        # Questo forza il modello a cercare conferme del perché è Phishing o Legit,
        # invece di provare a indovinare (evitando così allucinazioni).
        if is_phishing:
            instruction = (
                "You are a Cybersecurity Instructor. The following email is a PHISHING ATTACK. "
                "Explain briefly which elements (sender, urgency, links) make it suspicious."
            )
        else:
            instruction = (
                "You are a Cybersecurity Instructor. The following email is SAFE and LEGITIMATE. "
                "Explain briefly why it looks professional and safe (correct sender, no urgency)."
            )

        # 2. Prompt Alpaca modificato dinamicamente
        # Usiamo f-string per iniettare l'istruzione, e {{}} per lasciare il placeholder dell'input
        alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{{}}

### Response:
"""
        # 3. Prepariamo l'input
        # Tagliamo l'email a 800 caratteri per evitare errori di memoria o contesti troppo lunghi
        prompt = alpaca_prompt.format(email_body[:800])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 4. Generazione
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Risposta breve e concisa
                temperature=0.6,  # Creatività bilanciata per variare le frasi
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. Pulizia Output
        if "### Response:" in decoded:
            response = decoded.split("### Response:")[-1].strip()
        else:
            response = decoded

        return response