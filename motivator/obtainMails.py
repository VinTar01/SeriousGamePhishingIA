import json
import time
import os
import sys

# Gestione import per farlo funzionare sia da terminale che da IDE
try:
    from scripts.game_backend_v3_motivator import GameBackend
except ImportError:
    sys.path.append("scripts")
    from scripts.game_backend_v3_motivator import GameBackend


def collect_data(num_new_samples=150):
    """
    Genera nuove email e le aggiunge al dataset esistente.
    """
    filename = "dataset_with_expl_v2.json"
    dataset = []

    # 1. Se il file esiste già, carichiamo i dati vecchi
    if os.path.exists(filename):
        print(f"Trovato dataset esistente '{filename}'. Caricamento...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"   -> Recuperati {len(dataset)} esempi esistenti.")
        except json.JSONDecodeError:
            print("   -> Errore lettura JSON (file corrotto o vuoto). Si riparte da zero.")
            dataset = []
    else:
        print(f" Nessun dataset trovato. Ne creo uno nuovo: '{filename}'")

    # Calcoliamo l'ID di partenza (per non avere ID duplicati)
    start_id = len(dataset)

    # 2. Inizializziamo il Backend
    backend = GameBackend()
    backend.start_ai_loading()

    print("In attesa del caricamento modello Mistral...")
    while not backend.is_ready():
        time.sleep(2)

    print(f"Avvio generazione di {num_new_samples} NUOVI esempi...")

    target_skills = [0.30, 0.60, 0.95]

    for i in range(num_new_samples):
        # Ciclo skills
        forced_skill = target_skills[i % 3]
        backend.logic.current_skill = forced_skill

        print(f"   Generazione {i + 1}/{num_new_samples} (Skill: {forced_skill})...")
        email_data = backend.next_turn()

        label_str = "PHISHING" if backend.current_label == 1 else "LEGIT"
        level_name = backend.current_level

        # Creazione nuova entry con ID progressivo
        entry = {
            "id": start_id + i,
            "difficulty_level": level_name,
            "type": label_str,  # Questo campo è FONDAMENTALE per il tuo nuovo prompt
            "subject": email_data['subject'],
            "body": email_data['body'],
            # Questo campo sarà vuoto
            # prima di fare il training
            "explanation": ""
        }
        dataset.append(entry)

    # 3. Salvataggio finale (sovrascrive il file con la lista aggiornata)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Salvataggio completato in '{filename}'.")
    print(f"Totale esempi nel dataset: {len(dataset)} (Vecchi + Nuovi)")


if __name__ == "__main__":
    # Genera 60 nuove mail per volta (puoi cambiare il numero)
    collect_data(60)