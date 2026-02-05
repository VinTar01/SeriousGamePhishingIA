import os
import sys
import re
import random
import torch
import threading
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# =============================================================================
# GESTIONE IMPORT MOTIVATOR
# =============================================================================

#Eseguendo lo script da 'scripts/', Python non vede la cartella 'motivator' nella root.
# Soluzione: Aggiungo dinamicamente la directory genitore al sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ottiene la cartella genitore (SeriousGamePhishing2/)
parent_dir = os.path.dirname(current_dir)

# Aggiunge la root al path temporaneo per poter importare 'motivator'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Ora possiamo importare come se fossimo nella root.
    # MotivatorAI è la classe wrapper per TinyLlama
    from motivator.motivator import MotivatorAI
except ImportError:
    try:
        from motivator import MotivatorAI
    except ImportError:
        # se manca il modulo, il gioco funziona ma senza spiegazioni.
        print("ERRORE CRITICO: Il file 'motivator/motivator.py' non è stato trovato.")
        MotivatorAI = None

# =============================================================================
# 1. CONFIGURAZIONE
# =============================================================================


HF_TOKEN = "hf...."
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

ADAPTER_PATH = "C:/Users/daisl/Desktop/SeriousGamePhishing2/models/mistral_phishing_v3_final"

MOTIVATOR_PATH = "C:/Users/daisl/Desktop/SeriousGamePhishing2/motivator/models/tinyllama_motivator_v1"


# =============================================================================
# 2. GENERATORE EMAIL - sfrutta modello creato con train_mistral_v3.py
# =============================================================================

class PhishingEmailGenerator:
    """
    Gestisce il caricamento e l'inferenza del modello Mistral-7B.
    Include logiche di pulizia (regex) per rimuovere artefatti e PII dal dataset originale.
    """

    def __init__(self):
        self.model_loaded = False
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Carica il modello in 4-bit per risparmiare VRAM (QLoRA)."""
        print(">>> [AI ENGINE] Caricamento Modello LLM...")

        # Configurazione Quantizzazione 4-bit (NF4) essenziale per GPU consumer
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Carica il modello base "scheletro"
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",
            token=HF_TOKEN, trust_remote_code=True
        )
        # Inietta l'adapter LoRA addestrato specificamente per il phishing
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()  # Mette il modello in modalità inferenza (no dropout, ecc.)
        self.model_loaded = True
        print(">>> [AI ENGINE] Modello pronto.")

    def _sanitize_pii(self, text):
        """
        Rimuove dati sensibili o specifici del dataset di training (es. Enron).
        Serve per evitare bias cognitivi nell'utente e rendere l'email generica.
        """
        # Sostituisce email reali con placeholder
        email_pattern = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}\b'
        text = re.sub(email_pattern, "employee@target-company.com", text)

        # Rimuove riferimenti specifici al corpus Enron o Jose Nazario
        text = re.sub(r'\bEnron\b', "Corporate HQ", text, flags=re.IGNORECASE)
        text = re.sub(r'\bmonkey\.org\b', "suspicious-domain.net", text, flags=re.IGNORECASE)
        text = re.sub(r"dear\s+jose\b", "Dear Employee", text, flags=re.IGNORECASE)

        # Neutralizza i link per sicurezza (evita click accidentali su domini reali malevoli)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, "http://suspicious-link.com/login", text)

        # Aggiorna le date vecchie (dataset 2000-2005) all'anno corrente
        text = re.sub(r'\b(19|20)\d{2}\b', "2024", text)
        text = re.sub(r'/\s*00\b', "/ 24", text)
        text = re.sub(r'©\s*\d{4}.*', "", text)  # Rimuove copyright vecchi
        return text

    def _cut_loops(self, text):
        """
        Taglia le ripetizioni infinite (problema noto dei LLM su testi brevi).
        Se il modello ripete la stessa frase, tronca.
        """
        if len(text) < 30: return text
        snippet_len = 15
        start_snippet = text[:snippet_len]
        last_occ = text.rfind(start_snippet)
        if last_occ > 0:
            return text[:last_occ].strip()
        return text

    def _clean_artifacts(self, text, prompt_text=""):
        """
        Pipeline principale di pulizia. Rimuove:
        Es: 'Here is the email you asked for:', 'Subject:', 'Body:', ecc.
        """
        # Rimuove il prompt se il modello lo ripete all'inizio
        if prompt_text and text.startswith(prompt_text):
            text = text.replace(prompt_text, "", 1)

        # Rimuove token NaN o artefatti di debug
        text = re.sub(r"Subject:\s*nan", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Write a .*? email\.*\s*", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Lista di frasi di "meta-testo" da tagliare
        truncators = ["Examples:", "Avoid:", "Tips:", "Best regards, Your signature:", "Note that this is an example"]
        for t in truncators:
            if t in text: text = text.split(t)[0]

        # Regex per pulire inizi di frase non diegetici
        bad_starts = [r"^Based on the passage", r"^Optional:", r"^Note:", r"^This exercise aims",
                      r"^Can you paraphrase", r"^Here is a draft", r"^suspicious sender",
                      r"^- suspicious sender", r"^Attachments:", r"^Example: Subject:"]

        lines = text.split('\n')
        clean_lines = [line for line in lines if not any(re.search(p, line.strip(), re.IGNORECASE) for p in bad_starts)]
        text = "\n".join(clean_lines)

        # Separa Body da Subject se il modello li ha uniti esplicitamente
        if "Body:" in text:
            parts = re.split(r"Body:\s*", text, flags=re.IGNORECASE)
            if len(parts) > 1: text = parts[-1]

        patterns = [r"^Example \d+:", r"^Here is the email:", r"Subject:\s*$"]
        for p in patterns: text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

        text = text.strip()
        # Pulizia finale punteggiatura iniziale strana
        text = re.sub(r"^[\.\s\-\_\,]+", "", text)
        text = self._cut_loops(text)
        text = self._sanitize_pii(text)
        return text.strip()

    def _generate_subject_repair(self, body_text):
        """
        Funzione di Fallback: Se il modello genera un'email senza Oggetto,
        facciamo una seconda chiamata rapida all'IA per generare un oggetto basato sul corpo.
        """
        prompt = f"<s>[INST] Summarize the email below into a short Subject line.\nEmail Body: {body_text[:500]}...\nSubject: [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=0.5,
                                          pad_token_id=self.tokenizer.eos_token_id)
        subject = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return subject.replace("Subject:", "").replace('"', '').strip()

    def generate_email(self, user_prompt, temperature=0.75):
        """Metodo principale chiamato dal backend per ottenere il contenuto."""
        if not self.model_loaded:
            return {"subject": "Errore", "body": "Modello non ancora caricato."}

        # Prompting strutturato per Mistral [INST]
        full_prompt = f"<s>[INST] {user_prompt} [/INST]"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # Generazione (max 400 token per evitare email chilometriche)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=temperature,
                                          top_p=0.9, repetition_penalty=1.15, pad_token_id=self.tokenizer.eos_token_id)

        # Decoding e Pulizia
        raw_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        content = self._clean_artifacts(raw_output, prompt_text=user_prompt)

        # Tentativo di estrarre l'oggetto tramite regex
        subject_match = re.search(r"Subject:(.*?)\n", content, re.IGNORECASE)
        if subject_match:
            current_subject = subject_match.group(1).strip()
            body_text = re.sub(r"Subject:.*?\n", "", content, flags=re.IGNORECASE).strip()
        else:
            current_subject = ""
            body_text = content

        body_text = self._clean_artifacts(body_text)

        # Se l'oggetto manca o è rotto ("nan"), lo ripariamo
        if not current_subject or "nan" in current_subject.lower() or len(current_subject) < 3:
            current_subject = self._generate_subject_repair(body_text)

        return {"subject": current_subject, "body": body_text}


# =============================================================================
# 3. GESTORE DIFFICOLTÀ E VITE
# =============================================================================

class DifficultyManager:
    """
    Implementa la logica di gioco adattiva (Adaptive RL semplificato).
    Gestisce la Skill dell'utente, le vite rimaste e la scelta dello scenario.
    """

    def __init__(self):
        self.current_skill = 0.5  # Skill iniziale media
        self.learning_rate = 0.1  # Quanto velocemente cambia la skill
        self.profile_name = "Undefined"
        self.lives = 3  # Default
        self.max_lives = 3

    def set_profile(self, profile):
        """Imposta lo starting point basato sulla scelta utente Junior/Senior"""
        self.profile_name = profile.upper()
        if profile.lower() == "senior":
            # Senior: Parte da skill alta, ma ha meno vite
            self.current_skill = 0.75
            self.learning_rate = 0.05  # Skill più stabile
            self.lives = 2
            self.max_lives = 2
        else:  # Junior
            # Junior: Parte basso, impara più velocemente (LR alto), ha più margine di errore
            self.current_skill = 0.30
            self.learning_rate = 0.12
            self.lives = 4
            self.max_lives = 4
        print(f">>> [RL] Profilo: {self.profile_name} | Skill: {self.current_skill} | Vite: {self.lives}")

    def update_skill(self, user_won):
        """Aggiorna skill e vite dopo ogni risposta (Logica ELO-like)."""
        if user_won:
            # Guadagno decrescente: più sei bravo, più è difficile salire (1.1 - skill)
            gain = self.learning_rate * (1.1 - self.current_skill)
            self.current_skill += gain
            outcome = "WIN"
        else:
            # Penalità costante e perdita vita
            loss = self.learning_rate
            self.current_skill -= loss
            self.lives -= 1
            outcome = "LOSS"

        # Clamping della skill tra 0.1 e 1.0
        self.current_skill = max(0.1, min(1.0, self.current_skill))

        return outcome, self.current_skill, self.lives

    def get_next_scenario(self):
        """
        Determina il tipo di email da generare in base alla Skill attuale.
        Più alta è la skill, più è probabile ottenere phishing difficili o email legit ambigue.
        """
        skill = self.current_skill
        # Probabilità dinamica di ricevere una mail legittima
        chance_of_legit = 0.1 + (skill * 0.15)

        if random.random() < chance_of_legit:
            level_name = "LEGIT (Safe)"
            prompt = "Write a LEGIT corporate email regarding a scheduled meeting update or a project status. Use a professional tone."
            label = 0  # 0 = Legit
        else:
            label = 1  # 1 = Phishing
            # Soglie di difficoltà basate sulla skill
            #Prompt engeneering per creare ruoli specifici in base al livello di skill
            if skill < 0.45:
                level_name = "PHISHING EASY"
                # Prompt che forza errori grammaticali
                prompt = "Write a short PHISHING email demanding urgent and immediate password verification. Do not include instructions, just the email. Make grammatical mistakes, use some capital letters"
            elif skill < 0.80:
                level_name = "PHISHING HARD"
                # Prompt per email formali HR
                prompt = "Write a SOPHISTICATED PHISHING email regarding a policy update or benefits change. Use a formal HR tone."
            else:
                level_name = "PHISHING NIGHTMARE (HARD+)"
                # Spear Phishing: contesto realistico (reply, invoice)
                prompt = "Write a highly realistic PHISHING email mimicking a reply to a previous conversation about an invoice. Very subtle."

        return prompt, label, level_name


# =============================================================================
# 4. WRAPPER GIOCO
# =============================================================================

class GameBackend:
    """
    Classe Facade che unisce AI (Generazione), Motivator (Spiegazione) e Logica (Difficoltà).
    È l'unica interfaccia usata dal frontend  creato in main_v3_motivator.py.
    """

    def __init__(self):
        self.ai = PhishingEmailGenerator()
        self.motivator = None
        self.logic = DifficultyManager()
        self.current_email_data = None
        self.current_label = None
        self.current_level = ""

    def _load_task(self):
        """Task eseguito in un thread separato per non bloccare la UI all'avvio."""
        # 1. Carica il "mio" Mistral
        self.ai.load_model()

        # 2. Carica Motivator "mio" TinyLlama se disponibile
        if MotivatorAI:
            print(f">>> [GAME] Avvio caricamento Motivator da: {MOTIVATOR_PATH}")
            try:
                self.motivator = MotivatorAI(model_path=MOTIVATOR_PATH)
                print(">>> [GAME] Motivator collegato correttamente.")
            except Exception as e:
                print(f">>> [GAME] ERRORE caricamento Motivator: {e}")
                self.motivator = None
        else:
            print(">>> [GAME] ATTENZIONE: Motivator non disponibile.")

    def start_ai_loading(self):
        thread = threading.Thread(target=self._load_task)
        thread.start()

    def is_ready(self):
        return self.ai.model_loaded

    def set_profile(self, profile):
        self.logic.set_profile(profile)

    def next_turn(self):
        """Genera il prossimo turno di gioco."""
        prompt, label, level_name = self.logic.get_next_scenario()
        self.current_label = label
        self.current_level = level_name
        # Chiamata bloccante al modello per generare il testo
        self.current_email_data = self.ai.generate_email(prompt)
        return self.current_email_data

    def check_answer(self, user_says_phishing):
        """
        Valuta la risposta dell'utente, aggiorna il punteggio e genera la spiegazione.
        """
        user_val = 1 if user_says_phishing else 0
        is_correct = (user_val == self.current_label)

        # Aggiorna logica ELO
        outcome, new_skill, current_lives = self.logic.update_skill(is_correct)

        # Calcola stato del gioco (Vittoria/Sconfitta/Continua)
        game_status = "PLAYING"
        if current_lives <= 0:
            game_status = "LOSE"
        elif new_skill >= 1.0:
            game_status = "WIN"

        # Generazione spiegazione tramite Motivator
        explanation = "Analisi non disponibile."
        if self.motivator:
            try:
                # Passiamo il corpo dell'email e se era vera o falsa
                explanation = self.motivator.generate_explanation(
                    self.current_email_data['body'],
                    (self.current_label == 1)
                )
            except Exception as e:
                print(f"Errore generazione Motivator: {e}")
                explanation = "Errore nell'analisi AI."

        return {
            "correct": is_correct,
            "real_label": "PHISHING" if self.current_label == 1 else "LEGIT",
            "new_skill": new_skill,
            "lives": current_lives,
            "max_lives": self.logic.max_lives,
            "game_status": game_status,  # WIN, LOSE, PLAYING
            "level_played": self.current_level,
            "motivator": explanation
        }