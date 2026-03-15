import os
import sys
import re
import random
import torch
import threading
from scripts.game_logger import GameLogger
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# =============================================================================
# GESTIONE IMPORT MOTIVATOR
# =============================================================================

# Eseguendo lo script da 'scripts/', Python non vede la cartella 'motivator' nella root.
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


HF_TOKEN = "hf_..."
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

ADAPTER_PATH = "D:/SeriousGamePhishing2/models/mistral_phishing_v3_final"
MOTIVATOR_PATH = "D:/SeriousGamePhishing2/motivator/models/tinyllama_motivator_v2"


# =============================================================================
# 2. GENERATORE EMAIL E CONTROLLO MOTIVAZIONE - Mistral-7B
# =============================================================================

class PhishingEmailGenerator:
    """
    Gestisce il caricamento e l'inferenza del modello Mistral-7B.
    Genera le email e valuta le risposte testuali dell'utente.
    """

    def __init__(self):
        self.model_loaded = False
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(">>> [AI ENGINE] Caricamento Modello LLM...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",
            token=HF_TOKEN, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()
        self.model_loaded = True
        print(">>> [AI ENGINE] Modello pronto.")

    def _sanitize_pii(self, text):
        email_pattern = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}\b'
        text = re.sub(email_pattern, "employee@target-company.com", text)
        text = re.sub(r'\bEnron\b', "Corporate HQ", text, flags=re.IGNORECASE)
        text = re.sub(r'\bmonkey\.org\b', "suspicious-domain.net", text, flags=re.IGNORECASE)
        text = re.sub(r"dear\s+jose\b", "Dear Employee", text, flags=re.IGNORECASE)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, "http://suspicious-link.com/login", text)
        text = re.sub(r'\b(19|20)\d{2}\b', "2024", text)
        text = re.sub(r'/\s*00\b', "/ 24", text)
        text = re.sub(r'©\s*\d{4}.*', "", text)
        return text

    def _cut_loops(self, text):
        if len(text) < 30: return text
        snippet_len = 15
        start_snippet = text[:snippet_len]
        last_occ = text.rfind(start_snippet)
        if last_occ > 0:
            return text[:last_occ].strip()
        return text



    #Per gestire refusi dei prompt
    def _clean_artifacts(self, text, prompt_text=""):
        if prompt_text and text.startswith(prompt_text):
            text = text.replace(prompt_text, "", 1)

        # Rimuove convenevoli iniziali tipici degli LLM ("Sure, here is...", "Certainly!")
        text = re.sub(r"^(Sure|Certainly|Here is|Here's).*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)

        text = re.sub(r"Subject:\s*nan", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Write a .*? email\.*\s*", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Troncatori (taglia tutto quello che viene dopo queste parole)
        truncators = ["Examples:", "Avoid:", "Tips:", "Best regards, Your signature:", "Note that this is an example",
                      "---"]
        for t in truncators:
            if t in text: text = text.split(t)[0]

        # Rimuove intere righe che iniziano con queste parole (tipiche spiegazioni dell'IA)
        bad_starts = [
            r"^Based on the passage", r"^Optional:", r"^Note:", r"^This exercise aims",
            r"^Can you paraphrase", r"^Here is a draft", r"^suspicious sender",
            r"^- suspicious sender", r"^Attachments:", r"^Example: Subject:",
            r"^\**Example\**", r"^\**Email Example\**"
        ]

        lines = text.split('\n')
        clean_lines = [line for line in lines if not any(re.search(p, line.strip(), re.IGNORECASE) for p in bad_starts)]
        text = "\n".join(clean_lines)

        if "Body:" in text:
            parts = re.split(r"Body:\s*", text, flags=re.IGNORECASE)
            if len(parts) > 1: text = parts[-1]

        # Regex generiche per pulire la formattazione residua
        patterns = [
            r"^Example \d+:", r"^Here is the email:",
            r"^\**Subject:\**\s*$", r"^\**Subject\**\s*$"
        ]
        for p in patterns: text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

        text = text.strip()
        text = re.sub(r"^[\.\s\-\_\,]+", "", text)
        text = self._cut_loops(text)
        text = self._sanitize_pii(text)
        return text.strip()

    #Per gestire il problema dell'oggeto vuoto, se si verifica questo problema, lo genera chiamando il modello e passandogli il body
    def _generate_subject_repair(self, body_text):
        prompt = f"<s>[INST] Summarize the email below into a short Subject line.\nEmail Body: {body_text[:500]}...\nSubject: [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=0.5,
                                          pad_token_id=self.tokenizer.eos_token_id)
        subject = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return subject.replace("Subject:", "").replace('"', '').strip()

    #funzione per la generazione delle email con il modello addestrato
    def generate_email(self, user_prompt, temperature=0.75):
        if not self.model_loaded:
            return {"subject": "Errore", "body": "Modello non ancora caricato."}

        full_prompt = f"<s>[INST] {user_prompt} [/INST]"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=temperature,
                                          top_p=0.9, repetition_penalty=1.15, pad_token_id=self.tokenizer.eos_token_id)

        raw_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        content = self._clean_artifacts(raw_output, prompt_text=user_prompt)

        subject_match = re.search(r"Subject:(.*?)\n", content, re.IGNORECASE)
        if subject_match:
            current_subject = subject_match.group(1).strip()
            body_text = re.sub(r"Subject:.*?\n", "", content, flags=re.IGNORECASE).strip()
        else:
            current_subject = ""
            body_text = content

        body_text = self._clean_artifacts(body_text)

        if not current_subject or "nan" in current_subject.lower() or len(current_subject) < 3:
            current_subject = self._generate_subject_repair(body_text)

        return {"subject": current_subject, "body": body_text}


    #Funzione per gestire la valutazione della motivazione utente
    def evaluate_reasoning(self, email_body, is_phishing, user_reason):
        """
        Usa Mistral come giudice.
        Rubrica: 0 (vuota o nonsense), 1-3 (contraddizione/totalmente errata),
        4-6 (causa sbagliata/inventata), 7-10 (causa corretta).
        Include esempi completi per addestrare il modello in-context (Few-Shot).
        """
        user_reason = user_reason.strip()
        testo_lower = user_reason.lower()

        # 1. FILTRI PYTHON (Blocca la resa totale o l'assenza di input)
        resa_words = ["non lo so", "boh", "non so", "idk", "non saprei", "nessuna idea"]
        testo_pulito = user_reason.strip().lower()

        # Se la frase è cortissima o contiene palesi parole di resa
        if not user_reason or len(user_reason) < 6 or any(w in testo_pulito for w in resa_words):
            return 0.0

        # Se scrive l'esatta parola "niente" o "nessuna" (ma permette "nessuna minaccia")
        if testo_pulito in ["niente", "nessuna", "a caso", "casuale"]:
            return 0.0

        truth_label = "PHISHING" if is_phishing else "LEGITTIMA"

        # 2. FEW-SHOT PROMPTING
        prompt = (
            f"<s>[INST] You are a strict AI grader evaluating a student's explanation for classifying an email as {truth_label}.\n"
            f"SCORING RUBRIC (0-10):\n"
            f"- Score 0: GIBBERISH OR NONSENSE. The text is random letters ('asdfg'), keyboard mashing, completely unrelated chatter ('ciao'). YOU MUST GIVE EXACTLY 0. NEVER give 0 if the text discusses emails, security, or threats, even if completely wrong.\n"
            f"- Score 1-3: CONTRADICTION. The student writes a real, logical sentence, but it explains the EXACT OPPOSITE of {truth_label}. (e.g. {truth_label} is PHISHING but student says 'it seems safe', or vice-versa). DO NOT give 1-3 if the student just confused the company/brand name.\n"
            f"- Score 4-6: HALLUCINATION / BRAND MIX-UP / WRONG CAUSE. The explanation is decent but invents SPECIFIC PHYSICAL elements THAT DO NOT EXIST in the email text (e.g. mentions a malicious .mp4 or PDF attachment when there is ONLY a link, or swaps brand names like Amazon instead of Netflix). STRICTLY CAP AT 6 for these specific errors.\n"
            f"- Score 7-10: CORRECT AND LOGICAL. The explanation identifies at least one actual suspicious or legitimate element. IMPORTANT: Using general security deductions like 'stealing data', 'scam', 'fake link', or 'phishing' when they logically apply to the email's intent IS NOT A HALLUCINATION. Reward good logical deductions with 8, 9, or 10!\n\n"

            f"--- EXAMPLES ---\n"

            f"Context: The email is {truth_label}.\n"
            f"Student: \"asdfg qwerty ciao\"\n"
            f"Score: 0\n"
            f"(Reason: Complete nonsense/random words. Fits 0.)\n\n"
            
            f"Context: The email is {truth_label}.\n"
            f"Student: \"vfvfv swefkfeb v adfucjòdgvhbuiw\"\n"
            f"Score: 0\n"
            f"(Reason: Completely unrelated nonsense and keyboard mashing. Fits 0.)\n\n"

            f"Context: The email is LEGITTIMA.\n"
            f"Email: \"Ti confermo l'appuntamento di domani in sala riunioni B per parlare del progetto. Saluti.\"\n"
            f"Student: \"È una truffa, cerca di rubare i dati e c'è un link sospetto.\"\n"
            f"Score: 2\n"
            f"(Reason: Complete contradiction. The user clicked LEGITTIMA but described a PHISHING email. Fits 1-3 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"Il tuo account sta per scadere, clicca qui: http://fake-login.com\"\n"
            f"Student: \"L'email è del tutto sicura, non presenta minacce e il mittente sembra affidabile.\"\n"
            f"Score: 2\n"
            f"(Reason: Complete contradiction. The email is PHISHING but the student describes it as safe. Fits 1-3 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"Hai appena vinto una Gift Card da 500 euro! Clicca su questo link per riscattare il tuo premio subito.\"\n"
            f"Student: \"È la classica truffa della finta vincita per invogliarti a cliccare un link malevolo e rubarti i dati personali.\"\n"
            f"Score: 10\n"
            f"(Reason: Perfect logical deduction. Even if 'stealing data' isn't explicitly written in the email, it is the obvious intent of a gift card scam. This IS NOT a hallucination. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"Your Netflix subscription has been suspended. Update your payment details now!\"\n"
            f"Student: \"L'email presenta un tono urgente e minaccioso e chiede di cliccare un link per aggiornare i dati dell'account Amazon Prime Video.\"\n"
            f"Score: 5\n"
            f"(Reason: Brand Mix-up. The student correctly caught the urgency and the link, but hallucinated the brand 'Amazon Prime Video' instead of 'Netflix'. Fits 4-6 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"Gentile cliente, verifica subito la tua identità al link seguente: http://secure-update.com\"\n"
            f"Student: \"L'email contiene un file allegato .mp4 che in realtà è un malware per rubare i dati bancari.\"\n"
            f"Score: 4\n"
            f"(Reason: Severe Hallucination of a physical element. The student completely invented an .mp4 attachment and malware. The email ONLY contains a link. Fits 4-6 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"Il tuo account verrà sospeso tra 24 ore. Clicca qui http://login-finto.com per verificare la password.\"\n"
            f"Student: \"C'è troppa urgenza (24 ore) e l'URL non è quello ufficiale, in più chiede la password.\"\n"
            f"Score: 10\n"
            f"(Reason: Correctly identifies the urgency and the malicious link ACTUALLY present in the text. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is LEGITTIMA.\n"
            f"Email: \"Ecco il report mensile di cui parlavamo ieri. Ci aggiorniamo lunedì.\"\n"
            f"Student: \"Il tono è normale tra colleghi, non ci sono minacce né link strani e fa riferimento a conversazioni precedenti.\"\n"
            f"Score: 10\n"
            f"(Reason: Excellent analysis of a safe email. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is LEGITTIMA.\n"
            f"Email: \"[Insert standard corporate request/update regarding internal IT maintenance or HR procedures]\"\n"
            f"Student: \"L'email proviene da una fonte interna plausibile, utilizza un linguaggio puramente informativo e non richiede azioni urgenti o credenziali.\"\n"
            f"Score: 10\n"
            f"(Reason: Perfect analysis of a safe, standard corporate email. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING EASY: urgent threat about account suspension, bad grammar, obvious malicious link]\"\n"
            f"Student: \"Sfrutta il senso di urgenza minacciando la chiusura dell'account per farti cliccare su un link contraffatto.\"\n"
            f"Score: 10\n"
            f"(Reason: Correctly identifies the emotional manipulation (urgency) and the payload (fake link) typical of easy phishing. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING HARD: highly formal HR policy update with a disguised SharePoint/login link, NO urgency]\"\n"
            f"Student: \"Il tono è ingannevolmente formale e tranquillo, ma il link nasconde un portale di login falso per sottrarre le credenziali aziendali.\"\n"
            f"Score: 9\n"
            f"(Reason: Correctly identifies the subtle payload despite the lack of urgency. Fits 7-10 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING HARD: highly formal HR policy update with NO urgency and NO grammar errors, just a malicious link]\"\n"
            f"Student: \"È palesemente phishing perché è piena di errori grammaticali vergognosi e mi minaccia di licenziamento immediato se non clicco.\"\n"
            f"Score: 4\n"
            f"(Reason: Hallucination. The email is highly formal with NO errors and NO threats, but the student copy-pasted a generic phishing description. Fits 4-6 range.)\n\n"

            #Altri esempi per scenari simili ai prompt di generazione 

            f"Context: The email is LEGITTIMA.\n"
            f"Email: \"[Insert LEGIT: automated reminder from the HR system to submit monthly timesheets. Very short and standard.]\"\n"
            f"Student: \"È un normale promemoria automatico del sistema aziendale, non chiede password o dati strani e il tono è puramente informativo.\"\n"
            f"Score: 10\n"
            f"(Reason: Correctly identifies the automated and harmless nature of the legit HR timesheet reminder. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING EASY: short email about a failed package delivery asking to click a link to reschedule.]\"\n"
            f"Student: \"Si spaccia per un corriere e sfrutta l'ansia del pacco non consegnato per spingerti a cliccare sul finto link di tracciamento.\"\n"
            f"Score: 10\n"
            f"(Reason: Perfect analysis of the psychological trigger (package delay) and the payload (fake link). Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING EASY: scareware claiming the computer is infected with a virus and they must click a link to clean it.]\"\n"
            f"Student: \"L'email è sicura e mi è molto utile, devo cliccare per pulire il mio computer dal virus come dice l'antivirus.\"\n"
            f"Score: 2\n"
            f"(Reason: Complete contradiction. The email is a scareware PHISHING attempt, but the user believed the lie and called it safe/useful. Fits 1-3 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING HARD: from an automated e-signature service asking to review a standard NDA. Neutral tone, no threats.]\"\n"
            f"Student: \"Sembra una normale notifica per firmare un documento NDA, ma in realtà il link porta a una finta pagina di login per sottrarre le credenziali.\"\n"
            f"Score: 10\n"
            f"(Reason: Excellent deduction. The student caught the sophisticated payload hidden in a routine, threat-free DocuSign/NDA notification. Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING HARD: automated notification from Jira/Slack. Totally routine and automated, no urgency.]\"\n"
            f"Student: \"È palese che sia phishing perché il mittente mi insulta e mi chiede un riscatto in Bitcoin.\"\n"
            f"Score: 4\n"
            f"(Reason: Severe Hallucination. The Jira/Slack fake email is highly professional and routine, but the student hallucinated insults and a Bitcoin ransom. Fits 4-6 range.)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING NIGHTMARE: Spear phishing from CEO to HR asking for employee W-2 tax forms. Polite, authoritative, NO urgency.]\"\n"
            f"Student: \"Questo è Spear Phishing o CEO Fraud. Il truffatore finge di essere un dirigente per farsi inviare dati fiscali sensibili aggirando le normali procedure, anche senza usare link.\"\n"
            f"Score: 10\n"
            f"(Reason: Outstanding analysis of a payload-less social engineering attack (CEO Fraud / BEC). Fits 7-10 range. GIVE 10)\n\n"

            f"Context: The email is PHISHING.\n"
            f"Email: \"[Insert PHISHING NIGHTMARE: automated Microsoft Teams or Google Meet calendar invite with a fake meeting link.]\"\n"
            f"Student: \"Si nasconde dietro un normale invito a un meeting aziendale. Non c'è urgenza, ma il finto link della riunione nasconde una trappola.\"\n"
            f"Score: 10\n"
            f"(Reason: Perfect identification of a subtle Nightmare-level calendar threat. Fits 7-10 range. GIVE 10)\n\n"

            f"--- ACTUAL TASK ---\n"
            f"Context: The email is {truth_label}.\n"
            f"Email snippet: \"{email_body[:300]}...\"\n"
            f"Student: \"{user_reason}\"\n"
            f"Score: [/INST]"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        raw_eval = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Estrae il numero (se genera testo sporco tipo "Score: 8", prende solo "8")
        match = re.search(r'\d+', raw_eval)
        if match:
            score = int(match.group())
            score = min(score, 10)  # Sicurezza matematica per non sforare
            return score / 10.0

        return 0.0


# =============================================================================
# 3. GESTORE DIFFICOLTÀ E VITE
# =============================================================================

class DifficultyManager:
    """
    Logica adattiva con motivazione:
    Ora la Skill sale in base a QUANTO è corretta la motivazione dell'utente.
    """

    def __init__(self):
        self.current_skill = 0.5
        self.learning_rate = 0.1
        self.profile_name = "Undefined"
        self.lives = 3
        self.max_lives = 3

    def set_profile(self, profile):
        self.profile_name = profile.upper()
        if profile.lower() == "senior":
            self.current_skill = 0.75
            self.learning_rate = 0.05
            self.lives = 2
            self.max_lives = 2
        else:
            self.current_skill = 0.25
            self.learning_rate = 0.12
            self.lives = 4
            self.max_lives = 4
        print(f">>> [RL] Profilo: {self.profile_name} | Skill: {self.current_skill} | Vite: {self.lives}")

    def update_skill(self, click_correct, reason_score):
        """
        Calcola i punti: il guadagno decresce ad alti livelli.
        La motivazione aggiunge un bonus fisso per evitare stalli.
        """
        feedback_msg = ""

        if click_correct:
            # Calcolo base decrescente (es. a skill 0.90 prendi pochissimo)
            base_gain = self.learning_rate * (1.05 - self.current_skill)

            if reason_score >= 0.7:
                # 100% punti base + 0.04 BONUS FISSO
                gain = (base_gain * 1.0) + 0.04
                feedback_msg = "Corretto! Ottima analisi (+ Bonus punti)."
            elif reason_score >= 0.4:
                # 50% punti base + 0.02 BONUS FISSO
                gain = (base_gain * 0.5) + 0.02
                feedback_msg = "Corretto, ma motivazione imprecisa (+ Piccolo bonus)."
            else:
                # Contraddizione: 0 Punti totali
                gain = 0.0
                feedback_msg = "Click corretto per fortuna? Spiegazione in contraddizione. 0 Punti."

            self.current_skill += gain
            outcome = "WIN"
        else:
            # HA SBAGLIATO IL CLICK
            loss = self.learning_rate
            self.current_skill -= loss
            self.lives -= 1
            outcome = "LOSS"
            feedback_msg = "Classificazione errata. Leggi la spiegazione sotto."

        #Evita che la skill non esca mai dal range 0.1 - 1.0
        self.current_skill = max(0.1, min(1.0, self.current_skill))

        return outcome, self.current_skill, self.lives, feedback_msg

    # forniamo i prompt per generare in modo mirato email per il livello utente appropriato, previsti vari esempi
    def get_next_scenario(self):
        skill = self.current_skill
        chance_of_legit = 0.1 + (skill * 0.15)

        if random.random() < chance_of_legit:
            level_name = "LEGIT (Safe)"
            label = 0
            prompts = [
                "Write a LEGIT corporate email regarding a scheduled meeting update. Use a professional tone.",
                "Write a LEGIT internal email asking a colleague to review an attached draft document.",
                "Write a LEGIT IT department notification about upcoming weekend server maintenance. No action required.",
                "Write a LEGIT friendly email from a manager summarizing the weekly team goals."
                "Write a LEGIT corporate newsletter summarizing a recent company charity event or team building day.",
                "Write a LEGIT automated reminder from the HR system to submit monthly timesheets. Very short and standard.",
                "Write a LEGIT welcome email introducing a new employee to the department."
            ]
            prompt = random.choice(prompts)
        else:
            label = 1
            if skill < 0.45:
                level_name = "PHISHING EASY"
                prompts = [
                    "Write a short PHISHING email demanding urgent password verification. Make grammatical mistakes, use some capital letters.",
                    "Write a short PHISHING email about a failed package delivery. Ask to click a link to reschedule. Poor formatting.",
                    "Write a short PHISHING email claiming the user's mailbox is full and they must upgrade their quota immediately."
                    "Write a short PHISHING email claiming the user won a gift card or lottery.Very obvious scam, poor grammar, requests personal data.",
                    "Write a short PHISHING email mimicking a streaming service (like Netflix) saying the account is suspended due to payment failure. High urgency.",
                    "Write a short PHISHING scareware email claiming the user's computer is infected with a virus and they must click a link to clean it. Lots of exclamation marks."
                ]

                prompt = random.choice(prompts)
            elif skill < 0.80:
                level_name = "PHISHING HARD"
                prompts = [
                    "Write a SOPHISTICATED PHISHING email regarding a policy update or benefits change. Use a formal HR tone. DO NOT use urgent language or deadlines. Make it sound routine and boring.",
                    "Write a SOPHISTICATED PHISHING email impersonating the IT department, requiring a mandatory software update via a fake portal link. Tone must be calm and procedural, NO urgency.",
                    "Write a SOPHISTICATED PHISHING email sharing a fake 'Q3 Performance Review' document via OneDrive/SharePoint. Sound helpful and professional, absolutely NO threats or urgency.",
                    "Write a SOPHISTICATED PHISHING email from a Manager asking an employee to process a vendor payment. Use a casual, everyday corporate tone, NO urgency."
                    "Write a SOPHISTICATED PHISHING email acting as an automated notification from a tool like Jira, Slack or Salesforce. It says a colleague tagged the user. NO urgency, totally routine and automated.",
                    "Write a SOPHISTICATED PHISHING email from an automated e-signature service (like DocuSign or AdobeSign). It asks to review a standard Non-Disclosure Agreement (NDA). Neutral tone, NO threats, NO urgency.",
                    "Write a SOPHISTICATED PHISHING email from the finance department claiming an expense report was approved and providing a link to view the receipt. Casual, positive tone, NO urgency.",
                    "Write a SOPHISTICATED PHISHING email asking the employee to complete a mandatory 'annual compliance training' module via a provided link. Boring, standard corporate procedure."
                ]
                prompt = random.choice(prompts)
            else:
                level_name = "PHISHING NIGHTMARE (HARD+)"
                prompts = [
                    "Write a highly realistic PHISHING email mimicking a reply to a previous conversation about an invoice. Very subtle. The tone must be completely relaxed and normal. NO urgency.",
                    "Write a highly realistic BEC (Business Email Compromise) PHISHING email from a known supplier, requesting to change their bank account details for future payments. Professional and purely administrative.",
                    "Write a highly realistic PHISHING email impersonating a legal firm sending a notice of dispute. Use formal legal jargon, completely devoid of exclamation marks or immediate deadlines."
                    "Write a highly realistic SPEAR PHISHING email from a CEO or C-level executive to the HR department requesting employee W-2 tax forms for an external audit. Polite, authoritative, NO urgency.",
                    "Write a highly realistic PHISHING email disguised as an automated Microsoft Teams or Google Meet calendar invite for a 'Quarterly Sync'. It includes a fake meeting link. Extremely brief, looks like a system generated invite.",
                    "Write a highly realistic PHISHING email from a vendor replying to an ongoing thread with a link to a 'secure file drop' containing requested architectural plans or project scopes. Highly contextual and professional."
                ]
                prompt = random.choice(prompts)

        return prompt, label, level_name



# =============================================================================
# 4. WRAPPER GIOCO per far interagire la logica con il main
# =============================================================================

class GameBackend:
    def __init__(self):
        self.ai = PhishingEmailGenerator()
        self.motivator = None
        self.logic = DifficultyManager()
        self.logger = GameLogger(filename="game_data_export.csv", enabled=True)
        self.current_email_data = None
        self.current_label = None
        self.current_level = ""

    def _load_task(self):
        self.ai.load_model()
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
        self.logger.start_run(profile)

    def next_turn(self):
        """
        Genera il prossimo scenario.
        Implementa un controllo: se l'email generata è vuota o troppo corta,
        riprova a generarla (fino a 3 volte) in modo trasparente per l'utente.
        """
        max_retries = 3

        for attempt in range(max_retries):
            prompt, label, level_name = self.logic.get_next_scenario()
            self.current_label = label
            self.current_level = level_name
            self.current_email_data = self.ai.generate_email(prompt)

            # Se il corpo dell'email ha meno di 20 caratteri, è palesemente rotta/vuota.
            # Scartiamo e riproviamo.
            if len(self.current_email_data['body'].strip()) > 20:
                return self.current_email_data
            else:
                print(
                    f"[BACKEND] Email vuota rilevata (tentativo {attempt + 1}/{max_retries}). Rigenerazione in corso...")

        # Se fallisce 3 volte di fila, restituisce l'ultima generata (evita loop infiniti)
        return self.current_email_data

    #gestione della valutazione motivazione
    def check_answer(self, user_says_phishing, user_reason=""):
        """
        Valuta prima la validità della motivazione.
        Se è inaccettabile, blocca l'operazione. Altrimenti procede col punteggio.
        """
        user_reason = user_reason.strip()
        testo_lower = user_reason.lower()

        # 1. FILTRI VELOCI PYTHON (Risposte palesemente nulle o a caso)
        resa_words = ["non lo so", "boh", "non so", "idk", "non saprei", "nessuna idea"]
        testo_pulito = user_reason.strip().lower()

        if not user_reason or len(user_reason) < 4:
            return {"valid": False, "error_msg": "Motivazione troppo corta. Scrivi una frase di senso compiuto!"}

        if testo_pulito in ["niente", "nessuna", "a caso", "casuale"]:
            return {"valid": False, "error_msg": "Non è permesso arrendersi! Prova a ragionare sull'email."}

        if any(w in testo_pulito for w in resa_words):
            return {"valid": False, "error_msg": "Non è permesso arrendersi! Prova a ragionare sull'email."}

        # 2. CONTROLLO MISTRAL PREVENTIVO
        # Passiamo a Mistral la scelta dell'utente (user_says_phishing) invece della verità assoluta.
        # Mistral controllerà se la spiegazione HA SENSO per la scelta fatta e se non inventa cose inesistenti.
        reason_score = self.ai.evaluate_reasoning(
            self.current_email_data['body'],
            user_says_phishing,
            user_reason
        )

        #Blocchiamo una risposta con voto 0 (parole a caso o del tutto senza senso)
        if reason_score == 0.0:
            return {
                "valid": False,
                "error_msg": "Motivazione non valida (parole a caso o fuori contesto). Riprova!"
            }


        #se valida, procediamo aggiornando i punteggi
        is_actually_phishing = (self.current_label == 1)
        is_correct = (user_says_phishing == is_actually_phishing)

        #salvo la skill prima dell'aggiornamento
        skill_before = self.logic.current_skill

        outcome, new_skill, current_lives, feedback_msg = self.logic.update_skill(is_correct, reason_score)

        #calcolo del guadagno effettivo da passare al logger
        skill_gain = new_skill - skill_before

        # 4. Spiegazione Ufficiale (Motivator  TinyLlama FT)
        explanation = "Analisi non disponibile."
        if self.motivator:
            try:
                explanation = self.motivator.generate_explanation(
                    self.current_email_data['body'],
                    is_actually_phishing
                )
            except Exception as e:
                print(f"Errore generazione Motivator: {e}")
                explanation = "Errore nell'analisi AI."

        # 5. LOG E STATO DI GIOCO
        game_status = "PLAYING"
        if current_lives <= 0:
            game_status = "LOSE"
            self.logger.save_run()  #   Salva la partita se perdi
        elif new_skill >= 1.0:
            game_status = "WIN"
            self.logger.save_run()  #   Salva la partita se vinci

        #Registriamo i dati del turno in memoria
        self.logger.log_turn(
            level=self.current_level,
            subject=self.current_email_data.get('subject', 'No Subject'),
            email_body=self.current_email_data.get('body', 'No Body'),
            true_label=self.current_label,
            user_click=1 if user_says_phishing else 0,
            user_reason=user_reason,
            mistral_score=reason_score,
            skill_before=skill_before,
            skill_gain=skill_gain,
            skill_after=new_skill,
            outcome=outcome
        )

        #Salviamo fisicamente sul file CSV riga per riga
        self.logger.save_run()

        # 6. Restituiamo i feedback separati
        score_text = f" (Voto: {int(reason_score * 10)}/10)" if is_correct else ""
        final_feedback = f"{feedback_msg}{score_text}"

        return {
            "valid": True,  #Segnale per il Frontend che è tutto ok
            "correct": is_correct,
            "real_label": "PHISHING" if self.current_label == 1 else "LEGIT",
            "new_skill": new_skill,
            "lives": current_lives,
            "max_lives": self.logic.max_lives,
            "game_status": game_status,
            "level_played": self.current_level,
            "reason_score": reason_score,
            "feedback_msg": final_feedback,
            "motivator": explanation
        }