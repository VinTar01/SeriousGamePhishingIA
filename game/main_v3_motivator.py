import pygame
import sys
import threading
import textwrap

# GESTIONE IMPORT BACKEND
# Il main si trova nella root, ma il backend è in 'scripts/'.
# Devo gestire i path per essere sicuro che l'import funzioni sia da IDE che da terminale.
try:
    from scripts.game_backend_v3_motivator import GameBackend
except ImportError:
    try:
        from scripts.game_backend_v3_motivator import GameBackend
    except ImportError:
        # Fallback estremo: aggiungo manualmente la cartella al path
        sys.path.append("scripts")
        from scripts.game_backend_v3_motivator import GameBackend

# =============================================================================
# CONFIGURAZIONE GRAFICA
# =============================================================================
WIDTH, HEIGHT = 1024, 768
BG_COLOR = (30, 30, 30)  # Tema scuro dibackground
EMAIL_BG = (245, 245, 250)  # Sfondo chiaro per l'email
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
BLUE = (50, 100, 200)
GOLD = (255, 215, 0)
TEXT_COLOR = (20, 20, 20)
HIDDEN_COLOR = (40, 40, 40)  # Colore quasi invisibile per l'area segreta

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Phishing Hunter AI")

# Gestione Font: Provo a caricare font di sistema standard.
# Se il sistema non li ha (es. Linux server), uso il default di Pygame per evitare crash.
try:
    fonts = {
        "title": pygame.font.SysFont("Arial", 40, bold=True),
        "header": pygame.font.SysFont("Arial", 26, bold=True),
        "body": pygame.font.SysFont("Calibri", 22),  # Calibri è standard per le email
        "ui": pygame.font.SysFont("Arial", 20, bold=True),
        "small": pygame.font.SysFont("Consolas", 14),  # Font monospaziato per log/debug
        "big": pygame.font.SysFont("Arial", 60, bold=True)
    }
except:
    default_font = pygame.font.Font(None, 30)
    fonts = {k: default_font for k in ["title", "header", "body", "ui", "small", "big"]}


# =============================================================================
# CLASSE GESTIONE GIOCO
# =============================================================================

class PhishingGameApp:
    def __init__(self):
        # Istanzio il Backend (Facade Pattern).
        # Il frontend parla solo con questa classe.
        self.backend = GameBackend()

        # Avvio il caricamento dei modelli (Mistral + TinyLlama).
        # Nota: Non è bloccante qui, parte un thread nel backend.
        self.backend.start_ai_loading()

        self.state = "LOADING"  # per gestire le schermate
        self.email_data = {}
        self.result_data = {}
        self.loading_angle = 0
        self.final_game_status = "PLAYING"

        # ADMIN MODE / DEBUG
        # Non voglio mostrare skill e livelli agli utenti normali (rompe l'immersione).
        # Ho creato una variabile switch che attivo con un click segreto.
        self.show_debug = False
        # Area cliccabile segreta (in alto a sinistra, dove appare il testo "SYSTEM ONLINE")
        self.debug_rect = pygame.Rect(10, 10, 400, 40)

    def start_generation_thread(self, profile=None):
        """
        Gestisce la generazione dell'email.

        """

        def task():
            if profile:
                self.backend.set_profile(profile)
            # Chiamata bloccante al backend
            self.email_data = self.backend.next_turn()
            # Quando finisce, cambio stato. Il main loop se ne accorgerà al prossimo frame.
            self.state = "PLAY"

        self.state = "GENERATING"
        threading.Thread(target=task, daemon=True).start()

    def process_vote(self, is_phishing):
        """Invio la risposta dell'utente al backend e ricevo il feedback + motivazione."""
        self.result_data = self.backend.check_answer(user_says_phishing=is_phishing)
        # Controllo se il gioco è finito (Vittoria o Game Over)
        self.final_game_status = self.result_data.get("game_status", "PLAYING")
        self.state = "FEEDBACK"

    def reset_game(self):
        """Riavvia la partita tornando al menu principale."""
        self.state = "MENU"
        self.final_game_status = "PLAYING"
        self.show_debug = False  # Resetta la view admin al riavvio per sicurezza

    def handle_click(self, pos):
        """
        Gestisce i click che non sono pulsanti standard.
        Qui implemento l'EASTER EGG per attivare la modalità Admin/Debug.
        """
        if self.state == "PLAY":
            if self.debug_rect.collidepoint(pos):
                self.show_debug = not self.show_debug  # Toggle ON/OFF
                print(f"[UI] Debug Mode: {self.show_debug}")

    def draw_btn(self, txt, x, y, w, h, col):
        """Helper per disegnare bottoni con effetto hover."""
        rect = pygame.Rect(x, y, w, h)
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)
        # Schiarisco il colore se il mouse è sopra
        draw_col = (min(col[0] + 30, 255), min(col[1] + 30, 255), min(col[2] + 30, 255)) if hover else col
        pygame.draw.rect(screen, draw_col, rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, rect, width=2, border_radius=10)
        ts = fonts['ui'].render(txt, True, WHITE)
        screen.blit(ts, ts.get_rect(center=rect.center))
        # Ritorna True se cliccato
        return hover and pygame.mouse.get_pressed()[0]

    def draw_multiline(self, text, x, y, w, font, color):
        """
        Funzione essenziale: Pygame non supporta il testo a capo automatico.
        Uso 'textwrap' per spezzare il corpo dell'email in righe che stiano nel box.
        """
        line_height = font.get_linesize()
        paragraphs = text.split('\n')
        cur_y = y
        for p in paragraphs:
            # Calcolo approssimativo dei caratteri per riga basato sulla larghezza
            chars_per_line = int(w / 9)
            lines = textwrap.wrap(p, width=chars_per_line)
            for line in lines:
                if cur_y > HEIGHT - 150: break  # Evito di scrivere fuori schermo in basso
                surface = font.render(line, True, color)
                screen.blit(surface, (x, cur_y))
                cur_y += line_height
            cur_y += 10  # Spazio extra tra paragrafi

    def draw(self):
        """Main Render Loop: Disegna la schermata in base allo stato corrente."""
        screen.fill(BG_COLOR)

        # --- LOADING ---
        if self.state == "LOADING":
            # Animazione semplice (spinner rotante finto)
            self.loading_angle = (self.loading_angle + 5) % 360
            title = fonts['title'].render("CARICAMENTO SISTEMA AI...", True, WHITE)
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            sub = fonts['small'].render("Inizializzazione..", True, (150, 150, 150))
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40)))

            # Polling: chiedo al backend se i modelli sono pronti
            if self.backend.is_ready():
                self.state = "MENU"

        # --- MENU ---
        elif self.state == "MENU":
            t = fonts['title'].render("PHISHING HUNTER AI", True, WHITE)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, 100)))
            desc = fonts['body'].render("Scegli il livello di difficoltà", True, (200, 200, 200))
            screen.blit(desc, desc.get_rect(center=(WIDTH // 2, 160)))

            # Pulsanti Selezione Profilo
            if self.draw_btn("JUNIOR", WIDTH // 2 - 200, 300, 400, 70, BLUE):
                self.start_generation_thread("junior")

            if self.draw_btn("SENIOR", WIDTH // 2 - 200, 420, 400, 70, RED):
                self.start_generation_thread("senior")

        # --- GENERATING ---
        elif self.state == "GENERATING":
            # Schermata di attesa mentre il Thread AI lavora
            t = fonts['title'].render("GENERAZIONE SCENARIO...", True, WHITE)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

        # --- PLAY ---
        elif self.state == "PLAY":
            # --- HEADER INFO (ADMIN MODE) ---
            # Disegna il rettangolo "invisibile" per il toggle admin
            pygame.draw.rect(screen, HIDDEN_COLOR, self.debug_rect, border_radius=5)

            if self.show_debug:
                # ADMIN ATTIVO: Mostro i dati grezzi del backend (Livello skill, Label reale)
                lvl_txt = f"Livello: {self.backend.current_level}"
                skill_txt = f"Skill: {self.backend.logic.current_skill:.2f}"
                info_text = f"[ADMIN] {lvl_txt}  |  {skill_txt}"
                text_col = (100, 255, 100)  # Verde Matrix
            else:
                # ADMIN OFF: Testo generico per non spoilerare
                info_text = "SYSTEM ONLINE - Partita in corso"
                text_col = (80, 80, 80)  # Grigio scuro

            info_s = fonts['small'].render(info_text, True, text_col)
            screen.blit(info_s, info_s.get_rect(midleft=(20, 30)))

            # --- VITE (Sempre visibili) ---
            lives = self.backend.logic.lives
            max_lives = self.backend.logic.max_lives
            lives_txt = f"VITE: {lives}/{max_lives}"
            # Colore rosso se stai per morire (1 vita)
            live_col = RED if lives == 1 else GREEN
            lives_surf = fonts['header'].render(lives_txt, True, live_col)
            screen.blit(lives_surf, (WIDTH - 180, 20))

            # --- EMAIL CLIENT SIMULATION ---
            # Disegno il box bianco dell'email
            pygame.draw.rect(screen, EMAIL_BG, (50, 60, WIDTH - 100, HEIGHT - 180), border_radius=8)
            pygame.draw.line(screen, (200, 200, 200), (70, 110), (WIDTH - 70, 110), 2)  # Linea separatrice Subject/Body

            # Oggetto
            subj_lbl = fonts['header'].render(self.email_data.get('subject', 'No Subject'), True, TEXT_COLOR)
            screen.blit(subj_lbl, (70, 70))

            # Corpo Email (renderizzato riga per riga)
            self.draw_multiline(self.email_data.get('body', ''), 70, 130, WIDTH - 140, fonts['body'], TEXT_COLOR)

            # Bottoni Scelta
            if self.draw_btn("È LEGITTIMA", 100, HEIGHT - 100, 350, 60, GREEN):
                self.process_vote(is_phishing=False)
            if self.draw_btn("È PHISHING", WIDTH - 450, HEIGHT - 100, 350, 60, RED):
                self.process_vote(is_phishing=True)

        # --- FEEDBACK ---
        elif self.state == "FEEDBACK":
            # Recupero i dati calcolati dal backend
            is_correct = self.result_data.get('correct', False)
            real_label = self.result_data.get('real_label', 'UNKNOWN')
            new_skill = self.result_data.get('new_skill', 0.0)
            motivator_txt = self.result_data.get('motivator', '...')  # Spiegazione di TinyLlama
            lives = self.result_data.get('lives', 0)

            res_col = GREEN if is_correct else RED
            msg = "CORRETTO!" if is_correct else "SBAGLIATO!"

            t = fonts['title'].render(msg, True, res_col)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, 80)))

            # Box centrale per la spiegazione
            box_rect = pygame.Rect(WIDTH // 2 - 350, 150, 700, 350)
            pygame.draw.rect(screen, (50, 50, 60), box_rect, border_radius=15)
            pygame.draw.rect(screen, res_col, box_rect, width=2, border_radius=15)

            # Labels
            l1 = fonts['header'].render(f"Tipo Reale: {real_label}", True, WHITE)
            screen.blit(l1, l1.get_rect(center=(WIDTH // 2, 190)))

            l2 = fonts['ui'].render("Analisi Istruttore AI:", True, (150, 200, 255))
            screen.blit(l2, l2.get_rect(center=(WIDTH // 2, 240)))

            # Testo generato dal Motivator
            self.draw_multiline(motivator_txt, WIDTH // 2 - 320, 270, 640, fonts['body'], WHITE)

            # Info Progressioni
            prog_txt = f"Skill: {new_skill:.2f} | Vite Rimaste: {lives}"
            l3 = fonts['title'].render(prog_txt, True, WHITE)
            screen.blit(l3, l3.get_rect(center=(WIDTH // 2, 450)))

            # Pulsante "Prossimo" o "Risultato Finale"
            if self.final_game_status == "PLAYING":
                btn_txt = "PROSSIMA SFIDA >>"
                next_action = lambda: self.start_generation_thread(profile=None)
                btn_col = BLUE
            else:
                # Se abbiamo vinto o perso definitivamente, cambia il testo del bottone
                btn_txt = "VAI AL RISULTATO FINALE >>"
                next_action = lambda: setattr(self, 'state', "GAME_OVER")
                btn_col = GOLD if self.final_game_status == "WIN" else RED

            if self.draw_btn(btn_txt, WIDTH // 2 - 175, 580, 350, 70, btn_col):
                next_action()

        # --- GAME OVER ---
        elif self.state == "GAME_OVER":
            if self.final_game_status == "WIN":
                main_msg = "HAI VINTO!"
                sub_msg = "Congratulazioni! Hai raggiunto il livello massimo di Skill."
                color = GOLD
                btn_txt = "GIOCA ANCORA"
            else:
                main_msg = "HAI PERSO"
                sub_msg = "Hai esaurito tutte le vite disponibili."
                color = RED
                btn_txt = "RIAVVIA PARTITA"

            t = fonts['big'].render(main_msg, True, color)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))

            s = fonts['header'].render(sub_msg, True, WHITE)
            screen.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20)))

            if self.draw_btn(btn_txt, WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 80, BLUE):
                self.reset_game()


def main():
    clock = pygame.time.Clock()
    game = PhishingGameApp()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # GESTIONE CLICK SINGOLO PER TOGGLE ADMIN
            # Intercetto il click qui per verificare se ho cliccato nell'area segreta
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Tasto sinistro
                    game.handle_click(event.pos)

        game.draw()
        pygame.display.flip()
        clock.tick(30)  # Cap a 30 FPS max
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()