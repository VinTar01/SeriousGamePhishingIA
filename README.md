# Phishing Hunter AI

## Descrizione del Progetto
Phishing Hunter AI è un Serious Game adattivo sviluppato per la formazione degli utenti nel riconoscimento delle minacce di Phishing. Il sistema integra Large Language Models (LLM) per la generazione procedurale degli scenari e un algoritmo di Reinforcement Learning (ELO-like) per il bilanciamento dinamico della difficoltà. Il progetto è stato svolto utilizzando due modelli base distinti: Mistral-7B-Instruct-v0.3 per la generazione e la valutazione delle e-mail e TinyLlama-1.1B come "Motivator" per l'erogazione di feedback didattici in tempo reale. 

## Ambiente di Sviluppo Utilizzato
Il progetto è stato sviluppato, addestrato e testato utilizzando il seguente ambiente di lavoro:
* **Sistema Operativo:** Windows
* **Linguaggio:** Python 3.11
* **Hardware Principale:** GPU NVIDIA con supporto all'architettura CUDA
* **Librerie Fondamentali:** `torch`, `transformers`, `peft`, `bitsandbytes` (per l'addestramento e l'inferenza LLM), e `pygame` (per l'interfaccia grafica).
* **Servizi Esterni:** Account Hugging Face per il download dei pesi dei modelli base, i modelli scaricati non richiedono un token.

### Installazione delle Dipendenze
Per replicare l'ambiente di sviluppo, è consigliato l'utilizzo di un virtual environment. Le librerie impiegate possono essere installate tramite i seguenti comandi:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers peft bitsandbytes accelerate datasets
pip install pygame
```

## Architettura del Progetto e Descrizione degli Script



Di seguito viene dettagliata la funzione di ciascun file, specificandone i percorsi, gli input usati e gli output generati.



### 1. Preparazione dei Dati



#### `scripts/jsonConverter.py`

* **Descrizione:** Modulo di preprocessing dei dati primari. Estrae le informazioni dal dataset grezzo in formato CSV e le mappa all'interno di una struttura JSONL compatibile con le logiche di Instruction Tuning (formato Domanda-Risposta).

* **Input Utilizzato:** `dataset/raw/dataset_finale_ready_v3.csv`

* **Output Generato:** `dataset/processed/dataset_instruction.jsonl` (Pronto per l'addestramento di Mistral).



#### `motivator/obtainMails.py`

* **Descrizione:** Script dedicato alla generazione di e-mail sintetiche interrogando il modello Mistral dopo che quest'ultimo è stato sottoposto al fine-tuning. Le e-mail generate servono a creare la base dati per il Motivator e sono state successivamente completate (tramite LLM Gemini) con spiegazioni didattiche ad hoc.

* **Input Utilizzato:** Modello fine-tunato `models/mistral_phishing_v3_final`.

* **Output Intermedio:** `motivator/dataset_per_finetuning_motivator.json`.

* **Output Finale:** `motivator/dataset_with_expl_v2.json` (Dataset definitivo con le spiegazioni incluse).



---



### 2. Addestramento dei Modelli (Fine-Tuning)



#### `scripts/train_mistral_v3.py`

* **Descrizione:** Esegue il Supervised Fine-Tuning del modello Mistral-7B. Lo script implementa la tecnica QLoRA a 4-bit (NF4). Ha lo scopo di istruire il modello a generare e-mail procedurali mantenendo la netta distinzione tra contesto Legit e Phishing.

* **Input Utilizzato:** Il dataset processato `dataset/processed/dataset_instruction.jsonl`.

* **Output Generato:** Pesi e adattatori LoRA salvati nella directory locale `models/mistral_phishing_v3_final`.



#### `motivator/train_motivator.py`

* **Descrizione:** Script per il Fine-Tuning del modello compatto TinyLlama-1.1B. Utilizza il dataset generato per addestrare la rete a riconoscere specifici pattern di attacco e a produrre spiegazioni didattiche brevi e sintatticamente coerenti.

* **Input Utilizzato:** Il dataset contenente le spiegazioni `motivator/dataset_with_expl_v2.json`.
* **Output Generato:** Pesi e adattatori LoRA del Motivator salvati in `motivator/models/tinyllama_motivator_v2`.

---

### 3. Logica di Gioco e Motore AI (Backend)

#### `scripts/game_backend_v3_motivator.py`
* **Descrizione:** Gestisce la logica del gioco, richiamando sia il modello generatore (`mistral_phishing_v3_final`) sia il Motivator.
  * **PhishingEmailGenerator:** Gestisce il modello Mistral in locale. Costruisce i prompt dinamici, applica le regex di Post-Processing e agisce valutando da 0 a 10 la motivazione fornita dal giocatore.
  * **DifficultyManager:** Modulo di Reinforcement Learning. Imposta i profili (Junior/Senior), gestisce le vite, calcola algoritmicamente la probabilità di un'e-mail legittima e aggiorna la Skill con formula a rendimento decrescente.
* **Output:** Mantiene in memoria lo stato della sessione, fornisce gli scenari generati e richiama il sistema di tracciamento dati.

#### `motivator/motivator.py`
* **Descrizione:** Modulo contenente la classe wrapper `MotivatorAI`. Mette a disposizione il Motivator al backend e fornisce i prompt dedicati per interrogarlo. Gestisce il caricamento in memoria dei pesi di `tinyllama_motivator_v2` e si occupa dell'inferenza per produrre il feedback testuale.

---

### 4. Interfaccia Grafica (Frontend)

#### `game/main_v3_motivator.py`
* **Descrizione:** Utilizza il backend tramite la classe wrapper definita in `game_backend_v3_motivator.py`. Raccoglie i dati e renderizza il loop grafico: HUD, testi delle e-mail, barra della competenza, vite residue e caselle di inserimento testo. Intercetta gli input fisici dell'utente (click, tastiera) sincronizzandoli con le risposte dell'intelligenza artificiale.


## Istruzioni per la Replicabilità

Per eseguire una sessione di gioco, seguire i passaggi indicati:

1. **Configurazione Iniziale:** Aprire il file `game_backend_v3_motivator.py`. Controllare che le costanti `ADAPTER_PATH` e `MOTIVATOR_PATH` puntino correttamente alle cartelle dei modelli addestrati.
2. **Elaborazione Dati:** (Esclusivamente per ri-addestrare da zero) Eseguire in sequenza `obtainMails.py` e `jsonConverter.py`.
3. **Training:** (Esclusivamente per ri-addestrare da zero) Avviare i task di training tramite `python train_mistral_v3.py` e `python trainMotivator.py`.
4. **Avvio del Gioco:** Una volta completato il setup, avviare il gioco eseguendo:

```bash
python main_v3_motivator.py
```
L'interfaccia chiederà la selezione del profilo (Junior o Senior), dando il via al caricamento dei modelli nella VRAM e all'inizio della partita.

### 5. Istruzioni per la Replicabilità
Al termine di ogni iterazione utente, se l'utente lo ha selezionato tramite l'aposito pulsante nella homepage, il sistema effettua il salvataggio automatico dello stato all'interno del file game_data_export.csv. Il file raccoglie identificativi di sessione, livello di difficoltà, corpo dell'e-mail generata, motivazione fornita, valutazione AI e incremento della skill. Questo output strutturato è progettato per supportare future implementazioni.