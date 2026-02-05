
#script che simula 60 partite giocate, ottine le emails generate e le unisce in un json da completare tramite LLM con la descrizione dell'explanation
import json
import time
import random

from scripts.game_backend_v3 import GameBackend


def collect_data(num_samples=60):  # Aumentato leggermente per avere varietà
    backend = GameBackend()
    backend.start_ai_loading()

    print("In attesa del caricamento modello...")
    while not backend.is_ready():
        time.sleep(2)

    dataset = []

    print(f"Avvio generazione di {num_samples} esempi variegati...")

    # Definiamo 3 target di skill per coprire tutte le logiche del DifficultyManager
    # 0.20 -> Triggera PHISHING EASY
    # 0.60 -> Triggera PHISHING HARD
    # 0.95 -> Triggera PHISHING NIGHTMARE (e aumenta probabilità LEGIT)
    target_skills = [0.20, 0.60, 0.95]

    for i in range(num_samples):
        # 1. Cicla tra le difficoltà
        forced_skill = target_skills[i % 3]

        # 2. HACK: Forziamo direttamente la skill interna
        # Invece di usare set_profile, iniettiamo il valore numerico
        backend.logic.current_skill = forced_skill

        # Genera email (il backend userà la skill forzata per scegliere il prompt)
        print(f"Generazione {i + 1}/{num_samples} (Skill forzata: {forced_skill})...")
        email_data = backend.next_turn()

        # Recupera label reale
        # 0 = Legit, 1 = Phishing
        label_str = "PHISHING" if backend.current_label == 1 else "LEGIT"

        # Recupera il livello (utile per capire cosa ha scelto il backend)
        level_name = backend.current_level

        # Salvataggio entry
        entry = {
            "id": i,
            "difficulty_level": level_name,  # Utile per debug
            "type": label_str,
            "subject": email_data['subject'],
            "body": email_data['body'],
            "explanation": ""  # Da compilare con ChatGPT
        }
        dataset.append(entry)

    # Salvataggio su JSON
    filename = "dataset_per_finetuning_motivator.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ Dataset salvato in '{filename}'.")
    print(f"Generati {len(dataset)} esempi coprendo Easy, Hard e Nightmare.")


if __name__ == "__main__":
    collect_data(60)