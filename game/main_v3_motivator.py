import pygame
import sys
import threading
import textwrap

# Proviamo a importare il backend. Siccome la struttura delle cartelle
# può creare problemi se lo si avvia da posti diversi, usiamo dei try/except
try:
    from scripts.game_backend_v3_motivator import GameBackend
except ImportError:
    try:
        from scripts.game_backend_v3_motivator import GameBackend
    except ImportError:
        # Fallback estremo: forziamo l'aggiunta della cartella scripts
        sys.path.append("scripts")
        from scripts.game_backend_v3_motivator import GameBackend

# =============================================================================
# CONFIGURAZIONE GRAFICA E COLORI
# =============================================================================
WIDTH, HEIGHT = 1024, 768
BG_COLOR = (30, 30, 30)  # Sfondo scuro per far risaltare l'email
EMAIL_BG = (245, 245, 250)  # Sfondo grigio chiarissimo, stile client di posta
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
BLUE = (50, 100, 200)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)  # argento per il bonus "mediocre"
TEXT_COLOR = (20, 20, 20)
HIDDEN_COLOR = (40, 40, 40)  # Colore del tasto segreto per il debug

#finestra di gioco
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Phishing Hunter AI")

# Carichiamo i font. Se per caso non trova i font di sistema (es. su Linux/Mac),
# usa un font di default per evitare che il gioco crashi.
try:
    fonts = {
        "title": pygame.font.SysFont("Arial", 40, bold=True),
        "header": pygame.font.SysFont("Arial", 26, bold=True),
        "body": pygame.font.SysFont("Calibri", 22),
        "ui": pygame.font.SysFont("Arial", 20, bold=True),
        "small": pygame.font.SysFont("Consolas", 14),
        "big": pygame.font.SysFont("Arial", 60, bold=True)
    }
except:
    default_font = pygame.font.Font(None, 30)
    fonts = {k: default_font for k in ["title", "header", "body", "ui", "small", "big"]}


# =============================================================================
# UI COMPONENTS: La casella di testo dove l'utente scrive la motivazione
# ========================================================================
#creiamo una classe specifica per consentire cancellazione di piu caratteri
class InputBox:
    def __init__(self, x, y, w, h, font, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        # Colori per far capire se la casella è selezionata (attiva) o no
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.font = font
        self.txt_surface = self.font.render(text, True, TEXT_COLOR)
        self.active = False
        self.cursor_pos = len(self.text)

    def handle_event(self, event):
        # Gestione del click del mouse
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Se clicchi dentro il box, si attiva. Se clicchi fuori, si disattiva.
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive

        # Gestione della tastiera
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    pass  # Invio non fa nulla
                elif event.key == pygame.K_BACKSPACE:
                    if self.cursor_pos > 0:
                        self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                        self.cursor_pos -= 1
                elif event.key == pygame.K_DELETE:
                    if self.cursor_pos < len(self.text):
                        self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos + 1:]
                elif event.key == pygame.K_LEFT:
                    if self.cursor_pos > 0:
                        self.cursor_pos -= 1
                elif event.key == pygame.K_RIGHT:
                    if self.cursor_pos < len(self.text):
                        self.cursor_pos += 1
                else:
                    # Ignoriamo i tasti speciali (Tab, Esc, frecce su/giù, ecc.) per non sporcare il testo
                    if event.unicode and event.key not in [pygame.K_TAB, pygame.K_ESCAPE, pygame.K_UP, pygame.K_DOWN]:
                        #Limite di 250 CARATTERI
                        if len(self.text) < 250:
                            self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                            self.cursor_pos += 1

                # Ricrea la superficie grafica del testo ogni volta che l'utente preme o cancella qualcosa
                self.txt_surface = self.font.render(self.text, True, TEXT_COLOR)

    def clear(self):
        """Svuota la casella di testo e resetta correttamente il cursore."""
        self.text = ""
        self.cursor_pos = 0  #reset cursore
        self.txt_surface = self.font.render("", True, TEXT_COLOR)
        self.active = False
        self.color = self.color_inactive

    def draw(self, screen):
        # Disegniamo la base bianca e il bordino colorato
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)

        # --- GESTIONE SCORRIMENTO (SCROLLING) TESTO ---
        box_width = self.rect.width - 15  # Margine di sicurezza interno

        # Calcoliamo quanti pixel occupa la porzione di testo prima del cursore
        text_before_cursor = self.text[:self.cursor_pos]
        cursor_offset_x = self.font.size(text_before_cursor)[0]

        # Se il cursore va oltre il margine destro della casella, calcoliamo
        # di quanti pixel spostare l'intero blocco di testo verso sinistra.
        shift_x = 0
        if cursor_offset_x > box_width:
            shift_x = cursor_offset_x - box_width

        # Impostiamo una "maschera" grafica (clip) sulla casella:
        # tutto ciò che viene disegnato fuori da queste coordinate non sarà visibile.
        old_clip = screen.get_clip()
        screen.set_clip(self.rect)

        # Disegniamo il testo (spostato a sinistra se necessario)
        screen.blit(self.txt_surface, (self.rect.x + 5 - shift_x, self.rect.y + 8))

        # Disegniamo la stanghetta del cursore lampeggiante
        if self.active and pygame.time.get_ticks() % 1000 < 500:
            cursor_draw_x = self.rect.x + 5 + cursor_offset_x - shift_x
            pygame.draw.line(screen, TEXT_COLOR, (cursor_draw_x, self.rect.y + 8),
                             (cursor_draw_x, self.rect.y + self.rect.h - 8), 2)

        # Rimuoviamo la maschera per non bloccare i rendering successivi (pulsanti, ecc.)
        screen.set_clip(old_clip)

# =============================================================================
# LOGICA PRINCIPALE DEL GIOCO
# =============================================================================

class PhishingGameApp:
    def __init__(self):
        #Inizializza le AI (Mistral FT e Motivator) in background
        self.backend = GameBackend()
        self.backend.start_ai_loading()

        self.state = "LOADING"
        self.email_data = {}
        self.result_data = {}
        self.loading_angle = 0
        self.final_game_status = "PLAYING"

        #Impostazioni pre-partita di default
        self.selected_profile = "junior"
        self.record_csv = True

        #Casella per la motivazione
        self.input_box = InputBox(50, HEIGHT - 180, WIDTH - 100, 40, fonts['body'])

        #area cliccabile invisibile per vedere i info nascoste per debug
        self.show_debug = False
        self.debug_rect = pygame.Rect(10, 10, 400, 40)

        # Variabili per gestire quando l'utente scrive a caso e Mistral lo blocca
        self.input_error_msg = ""
        self.input_error_time = 0

    def start_generation_thread(self, profile=None):
        # Usiamo un thread separato per chiedere a Mistral la nuova email.
        # Così l'interfaccia grafica non si congela ("non risponde") durante l'attesa!
        def task():
            if profile:
                self.backend.set_profile(profile)
            self.email_data = self.backend.next_turn()
            self.state = "PLAY"

        # Svuotiamo e resettiamo l'input box
        self.input_box.clear()

        self.state = "GENERATING"
        threading.Thread(target=task, daemon=True).start()


    def process_vote(self, is_phishing, reason):
        """Prende la scelta e la frase dell'utente e le manda al giudice IA"""
        self.input_error_msg = ""  # Resetta vecchi errori a schermo

        # Chiamata al backend
        try:
            self.result_data = self.backend.check_answer(user_says_phishing=is_phishing, user_reason=reason)
        except TypeError:
            self.result_data = self.backend.check_answer(user_says_phishing=is_phishing)

        #Mistral ha detto che la frase non ha senso? (voto 0.0)
        if not self.result_data.get("valid", True):
            # Coloriamo il box di rosso e prepariamo il messaggio di errore.
            self.input_box.color = RED
            self.input_box.active = True
            self.input_error_msg = self.result_data.get("error_msg", "Input non valido. Riprova.")
            self.input_error_time = pygame.time.get_ticks()
            return  #Il turno non viene giocato.

        # SE LA RISPOSTA È VALIDA:
        # Pulisco il box e il cursore in un colpo solo
        self.input_box.clear()

        self.final_game_status = self.result_data.get("game_status", "PLAYING")
        self.state = "FEEDBACK"

    def reset_game(self):
        """Riporta tutto al menu iniziale dopo una sconfitta o vittoria"""
        self.state = "MENU"
        self.final_game_status = "PLAYING"
        self.show_debug = False

    def handle_click(self, pos):
        """Gestisce click speciali, come attivare il debug mode"""
        if self.state == "PLAY":
            if self.debug_rect.collidepoint(pos):
                self.show_debug = not self.show_debug

    def draw_btn(self, txt, x, y, w, h, col):
        """Funzione per disegnare bottoni col bordo smussato e l'effetto hover"""
        rect = pygame.Rect(x, y, w, h)
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)

        # Se c'è il mouse sopra, schiarisce il colore di base
        draw_col = (min(col[0] + 30, 255), min(col[1] + 30, 255), min(col[2] + 30, 255)) if hover else col

        pygame.draw.rect(screen, draw_col, rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, rect, width=2, border_radius=10)

        ts = fonts['ui'].render(txt, True, WHITE)
        screen.blit(ts, ts.get_rect(center=rect.center))

        # Restituisce True solo nel momento in cui viene premuto
        return hover and pygame.mouse.get_pressed()[0]

    def draw_multiline(self, text, x, y, w, font, color):
        """Permette a Pygame di andare a capo in automatico se il testo è troppo lungo"""
        line_height = font.get_linesize()
        paragraphs = text.split('\n')
        cur_y = y
        for p in paragraphs:
            chars_per_line = int(w / 9)  # Una stima di quanti caratteri entrano nella larghezza
            lines = textwrap.wrap(p, width=chars_per_line)
            for line in lines:
                if cur_y > HEIGHT - 300: break  # Taglia il testo per non coprire i bottoni in basso
                surface = font.render(line, True, color)
                screen.blit(surface, (x, cur_y))
                cur_y += line_height
            cur_y += 10

    def draw(self):
        """Il cuore pulsante della grafica. Viene richiamato 30 volte al secondo."""
        screen.fill(BG_COLOR)

        # --- SCHERMATA DI CARICAMENTO ---
        if self.state == "LOADING":
            self.loading_angle = (self.loading_angle + 5) % 360
            title = fonts['title'].render("CARICAMENTO SISTEMA AI...", True, WHITE)
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            sub = fonts['small'].render("Inizializzazione dei pesi neurali...", True, (150, 150, 150))
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40)))

            if self.backend.is_ready():
                self.state = "MENU"

        # --- SCHERMATA MENU PRE-PARTITA ---
        elif self.state == "MENU":
            t = fonts['title'].render("PHISHING HUNTER AI", True, WHITE)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, 100)))
            desc = fonts['body'].render("Imposta le regole prima di iniziare", True, (200, 200, 200))
            screen.blit(desc, desc.get_rect(center=(WIDTH // 2, 150)))

            # SEZIONE 1: Scelta difficoltà
            lbl_prof = fonts['ui'].render("1. Livello di difficoltà:", True, WHITE)
            screen.blit(lbl_prof, lbl_prof.get_rect(center=(WIDTH // 2, 230)))

            # Grigiamo il bottone non selezionato per far capire quale è attivo
            j_col = BLUE if self.selected_profile == "junior" else (80, 80, 80)
            s_col = RED if self.selected_profile == "senior" else (80, 80, 80)

            if self.draw_btn("JUNIOR", WIDTH // 2 - 220, 270, 200, 60, j_col):
                self.selected_profile = "junior"
            if self.draw_btn("SENIOR", WIDTH // 2 + 20, 270, 200, 60, s_col):
                self.selected_profile = "senior"

            # SEZIONE 2: Registrazione dei dati
            toggle_y = 410
            lbl_csv = fonts['ui'].render("2. Registra storico partita (CSV):", True, WHITE)
            screen.blit(lbl_csv, lbl_csv.get_rect(midright=(WIDTH // 2 + 10, toggle_y)))

            circle_center = (WIDTH // 2 + 40, toggle_y)
            circle_radius = 16

            mouse_pos = pygame.mouse.get_pos()
            dist = ((mouse_pos[0] - circle_center[0]) ** 2 + (mouse_pos[1] - circle_center[1]) ** 2) ** 0.5
            is_hover = dist <= circle_radius

            base_col = GREEN if self.record_csv else RED
            draw_col = (min(base_col[0] + 50, 255), min(base_col[1] + 50, 255),
                        min(base_col[2] + 50, 255)) if is_hover else base_col

            pygame.draw.circle(screen, draw_col, circle_center, circle_radius)
            pygame.draw.circle(screen, WHITE, circle_center, circle_radius, 2)
            status_txt = fonts['ui'].render("ON" if self.record_csv else "OFF", True, base_col)
            screen.blit(status_txt, (circle_center[0] + 25, toggle_y - 12))

            if is_hover and pygame.mouse.get_pressed()[0]:
                self.record_csv = not self.record_csv
                pygame.time.wait(200)  # Evita l'effetto stroboscopico del doppio click

            # SEZIONE 3: Avvio
            if self.draw_btn("AVVIA PARTITA", WIDTH // 2 - 150, 520, 300, 70, GREEN):
                # Passiamo la scelta al logger del backend
                if hasattr(self.backend, 'logger'):
                    self.backend.logger.enabled = self.record_csv
                self.start_generation_thread(self.selected_profile)

        # --- SCHERMATA DI ATTESA GENERAZIONE ---
        elif self.state == "GENERATING":
            t = fonts['title'].render("L'Hacker sta scrivendo l'email...", True, WHITE)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

        # --- FASE DI GIOCO ATTIVA ---
        elif self.state == "PLAY":
            pygame.draw.rect(screen, HIDDEN_COLOR, self.debug_rect, border_radius=5)

            # Debug Overlay (Visibile solo se si clicca in alto a sinistra)
            if self.show_debug:
                lvl_txt = f"Livello: {self.backend.current_level}"
                skill_txt = f"Skill: {self.backend.logic.current_skill:.2f}"
                info_text = f"[ADMIN] {lvl_txt}  |  {skill_txt}"
                text_col = (100, 255, 100)
            else:
                info_text = "SYSTEM ONLINE - Identifica la minaccia"
                text_col = (80, 80, 80)

            info_s = fonts['small'].render(info_text, True, text_col)
            screen.blit(info_s, info_s.get_rect(midleft=(20, 30)))

            # Disegno Vite
            lives = self.backend.logic.lives
            max_lives = self.backend.logic.max_lives
            lives_txt = f"VITE: {lives}/{max_lives}"
            live_col = RED if lives == 1 else GREEN
            lives_surf = fonts['header'].render(lives_txt, True, live_col)
            screen.blit(lives_surf, (WIDTH - 180, 20))

            # Disegno layout della finta Email
            pygame.draw.rect(screen, EMAIL_BG, (50, 60, WIDTH - 100, HEIGHT - 300), border_radius=8)
            pygame.draw.line(screen, (200, 200, 200), (70, 110), (WIDTH - 70, 110), 2)

            subj_lbl = fonts['header'].render(self.email_data.get('subject', 'No Subject'), True, TEXT_COLOR)
            screen.blit(subj_lbl, (70, 70))
            self.draw_multiline(self.email_data.get('body', ''), 70, 130, WIDTH - 140, fonts['body'], TEXT_COLOR)

            # Input Motivazione
            lbl_reason = fonts['ui'].render("Spiega brevemente la tua scelta (Obbligatorio):", True, WHITE)
            screen.blit(lbl_reason, (50, HEIGHT - 215))

            self.input_box.draw(screen)

            # Gestione popup di errore se l'utente ha scritto cose a caso (Mistral Voto 0)
            if self.input_error_msg:
                # Appare per 5 secondi per far capire lo sbaglio
                if pygame.time.get_ticks() - self.input_error_time < 5000:
                    err_surf = fonts['ui'].render(self.input_error_msg, True, RED)
                    screen.blit(err_surf, (50, HEIGHT - 135))
                else:
                    self.input_error_msg = ""
                    self.input_box.color = self.input_box.color_inactive

            # I due pulsantoni vitali
            if self.draw_btn("È LEGITTIMA", 100, HEIGHT - 100, 350, 60, GREEN):
                self.process_vote(is_phishing=False, reason=self.input_box.text)
            if self.draw_btn("È PHISHING", WIDTH - 450, HEIGHT - 100, 350, 60, RED):
                self.process_vote(is_phishing=True, reason=self.input_box.text)

        # --- SCHERMATA FEEDBACK ---
        elif self.state == "FEEDBACK":
            is_correct = self.result_data.get('correct', False)
            real_label = self.result_data.get('real_label', 'UNKNOWN')
            new_skill = self.result_data.get('new_skill', 0.0)
            reason_score = self.result_data.get('reason_score', 0.0)

            feedback_txt = self.result_data.get('feedback_msg', 'Feedback non disponibile.')
            motivator_txt = self.result_data.get('motivator', 'Spiegazione non disponibile.')
            lives = self.result_data.get('lives', 0)

            # Scritta principale risultato
            res_col = GREEN if is_correct else RED
            msg = "CORRETTO!" if is_correct else "SBAGLIATO!"
            t = fonts['title'].render(msg, True, res_col)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, 50)))

            # BOX 1: Il voto che ti ha dato Mistral sulla tua spiegazione
            box1_rect = pygame.Rect(WIDTH // 2 - 350, 90, 700, 110)
            pygame.draw.rect(screen, (40, 45, 50), box1_rect, border_radius=10)
            pygame.draw.rect(screen, res_col, box1_rect, width=2, border_radius=10)

            lbl_fdbk = fonts['ui'].render("Valutazione tua risposta:", True, GOLD if is_correct else RED)
            screen.blit(lbl_fdbk, (WIDTH // 2 - 330, 100))
            self.draw_multiline(feedback_txt, WIDTH // 2 - 330, 130, 660, fonts['body'], WHITE)

            # BOX 2: Spiegazione ufficiale dell'istruttore AI
            box2_rect = pygame.Rect(WIDTH // 2 - 350, 220, 700, 310)
            pygame.draw.rect(screen, (30, 35, 45), box2_rect, border_radius=10)
            pygame.draw.rect(screen, BLUE, box2_rect, width=2, border_radius=10)

            l1 = fonts['header'].render(f"La verità: l'email era {real_label}", True, WHITE)
            screen.blit(l1, (WIDTH // 2 - 330, 230))

            l2 = fonts['ui'].render("Analisi dell'Istruttore AI:", True, (150, 200, 255))
            screen.blit(l2, (WIDTH // 2 - 330, 270))

            self.draw_multiline(motivator_txt, WIDTH // 2 - 330, 300, 660, fonts['body'], (220, 220, 220))

            #MOSTRA SKILL E BONUS IN MODO ESPLICITO
            # 1. Punteggio Base
            prog_txt = f"Skill Attuale: {new_skill:.2f}   |   Vite Rimaste: {lives}"
            l3 = fonts['header'].render(prog_txt, True, WHITE)
            screen.blit(l3, l3.get_rect(center=(WIDTH // 2, 550)))

            # 2. Esplicitazione del Bonus Ragionamento
            if is_correct:
                if reason_score >= 0.7:
                    bonus_txt = "+ BONUS MAX RAGIONAMENTO (+0.04)"
                    bonus_col = GOLD
                elif reason_score >= 0.4:
                    bonus_txt = "+ BONUS MIN RAGIONAMENTO (+0.02)"
                    bonus_col = SILVER
                else:
                    bonus_txt = "NESSUN BONUS (Hai indovinato ma spiegato male!)"
                    bonus_col = RED
            else:
                bonus_txt = "- PENALITÀ (Hai perso una vita)"
                bonus_col = RED

            l_bonus = fonts['ui'].render(bonus_txt, True, bonus_col)
            screen.blit(l_bonus, l_bonus.get_rect(center=(WIDTH // 2, 580)))

            # Pulsante per andare avanti
            if self.final_game_status == "PLAYING":
                btn_txt = "PROSSIMA SFIDA >>"
                next_action = lambda: self.start_generation_thread(profile=None)
                btn_col = BLUE
            else:
                btn_txt = "VAI AL RISULTATO FINALE >>"
                next_action = lambda: setattr(self, 'state', "GAME_OVER")
                btn_col = GOLD if self.final_game_status == "WIN" else RED

            if self.draw_btn(btn_txt, WIDTH // 2 - 175, 620, 350, 70, btn_col):
                next_action()

        # --- SCHERMATA FINALE ---
        elif self.state == "GAME_OVER":
            if self.final_game_status == "WIN":
                main_msg = "HAI VINTO!"
                sub_msg = "Sei un esperto! Hai riconosciuto tutte le trappole (Skill = 1.0)."
                color = GOLD
                btn_txt = "GIOCA ANCORA"
            else:
                main_msg = "HAI PERSO"
                sub_msg = "L'azienda è stata hackerata. Hai finito le vite."
                color = RED
                btn_txt = "RIAVVIA PARTITA"

            t = fonts['big'].render(main_msg, True, color)
            screen.blit(t, t.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80)))

            s = fonts['header'].render(sub_msg, True, WHITE)
            screen.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 10)))

            #conferma il salvataggio della partita
            if self.record_csv:
                csv_msg = fonts['ui'].render("Dati della partita salvati in: game_data_export.csv", True, GREEN)
                screen.blit(csv_msg, csv_msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40)))

            if self.draw_btn(btn_txt, WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 80, BLUE):
                self.reset_game()


def main():
    clock = pygame.time.Clock()
    game = PhishingGameApp()
    running = True

    # Abilita la ripetizione dei tasti tenuti premuti

    pygame.key.set_repeat(300, 50)


    click_cooldown = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Se siamo nella schermata di gioco, passiamo l'input della tastiera al box della motivazione
            if game.state == "PLAY":
                game.input_box.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and click_cooldown == 0:
                    game.handle_click(event.pos)
                    click_cooldown = 10  # Blocca altri click per 10 frame

        if click_cooldown > 0:
            click_cooldown -= 1

        game.draw()
        pygame.display.flip()
        clock.tick(30)  # Il gioco gira fisso a 30 FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()