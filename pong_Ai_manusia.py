import pygame
import torch
from pong_env import PongEnv
from dqn_model import DQN
from mcts_agent import MCTSAgent

# Fungsi menggambar tombol
def draw_button(text, rect, font, screen, color=(50, 200, 50), text_color=(255, 255, 255)):
    pygame.draw.rect(screen, color, rect)
    label = font.render(text, True, text_color)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

# Inisialisasi pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Pong: Human vs AI")
clock = pygame.time.Clock()

# Environment dengan player manual
env = PongEnv(player_controlled=True)
model = DQN(6, 3)
model.load_state_dict(torch.load("pong_dqn.pth"))
model.eval()
ai_agent = MCTSAgent(model, simulations=50)

# Skor awal
human_score = 0
ai_score = 0

# Font dan tombol
font = pygame.font.SysFont("Arial", 28)
button_rect = pygame.Rect(env.width//2 - 80, env.height//2 + 40, 160, 50)

# Game state
state = env.reset()
running = True
show_result = False
result_timer = 0
winner_text = ""
game_over = False

while running:
    pygame.event.pump()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Kontrol player manual (W/S)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        env.player_y -= env.paddle_speed
    elif keys[pygame.K_s]:
        env.player_y += env.paddle_speed
    env.player_y = max(0, min(env.height - env.paddle_height, env.player_y))

    # AI ambil aksi
    ai_action = ai_agent.select_action(env, state)

    # Jalankan step
    state, reward, done, _ = env.step(ai_action)

    # Render arena
    env.render(screen)

    # Tampilkan skor
    score_text = font.render(f"Player: {ai_score}   AI: {human_score}", True, (255, 255, 0))
    screen.blit(score_text, (env.width // 2 - score_text.get_width() // 2, 10))

    # Tampilkan hasil poin
    if done and not show_result:
        if reward > 0:
            human_score += 1
            winner_text = "AI Mendapat Poin!"
        else:
            ai_score += 1
            winner_text = "Player Mendapat Poin!"

        # Periksa akhir permainan
        if ai_score >= 10:
            winner_text = "PLAYER MENANG!"
            game_over = True
        elif human_score >= 10:
            winner_text = "AI MENANG!"
            game_over = True

        result_timer = pygame.time.get_ticks()
        show_result = True

    if show_result:
        text = font.render(winner_text, True, (255, 0, 0))
        screen.blit(text, (env.width // 2 - text.get_width() // 2, 50))

        # Jika permainan selesai, tampilkan tombol
        if game_over:
            draw_button("Play Again", button_rect, font, screen)
            pygame.display.flip()

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect.collidepoint(event.pos):
                            # Reset semuanya
                            ai_score = 0
                            human_score = 0
                            state = env.reset()
                            game_over = False
                            show_result = False
                            waiting = False
        else:
            # Jika bukan game over, tunggu transisi 1.5 detik
            if pygame.time.get_ticks() - result_timer > 1500:
                state = env.reset()
                show_result = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
