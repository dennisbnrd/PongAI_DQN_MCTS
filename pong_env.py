import numpy as np
import pygame

class PongEnv:
    def __init__(self, width=640, height=480, paddle_speed=10, ball_speed=10, player_controlled=False):
        self.width = width
        self.height = height
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed
        self.player_controlled = player_controlled

        self.paddle_width = 10
        self.paddle_height = 80
        self.ball_size = 10

        self.reset()

    def reset(self):
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.ai_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vel_x = np.random.choice([-1, 1]) * self.ball_speed
        self.ball_vel_y = np.random.uniform(-1, 1) * self.ball_speed
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.player_y / self.height,
            self.ai_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_vel_x / self.ball_speed,
            self.ball_vel_y / self.ball_speed
        ], dtype=np.float32)

    def step(self, ai_action):
        if self.done:
            return self._get_state(), np.float32(0), True, {}

        # Gerakan AI
        if ai_action == 0:
            self.ai_y -= self.paddle_speed
        elif ai_action == 1:
            self.ai_y += self.paddle_speed
        self.ai_y = np.clip(self.ai_y, 0, self.height - self.paddle_height)

        # Gerakan otomatis player (jika tidak manual)
        if not self.player_controlled:
            if self.player_y + self.paddle_height / 2 < self.ball_y:
                self.player_y += self.paddle_speed
            elif self.player_y + self.paddle_height / 2 > self.ball_y:
                self.player_y -= self.paddle_speed
            self.player_y = np.clip(self.player_y, 0, self.height - self.paddle_height)

        # Update bola
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_vel_y *= -1

        reward = np.float32(0)

        if self.ball_x <= self.paddle_width and self.player_y < self.ball_y < self.player_y + self.paddle_height:
            self.ball_vel_x *= -1
        elif self.ball_x >= self.width - self.paddle_width - self.ball_size and self.ai_y < self.ball_y < self.ai_y + self.paddle_height:
            self.ball_vel_x *= -1
            reward = np.float32(0.5)
        elif self.ball_x >= self.width:
            reward = np.float32(-1)
            self.done = True
        elif self.ball_x <= 0:
            reward = np.float32(1)
            self.done = True
        else:
            reward = np.float32(-0.01 if ai_action == 2 else 0)

        ai_center = self.ai_y + self.paddle_height / 2
        distance = abs(ai_center - self.ball_y) / self.height
        reward = np.float32(reward + (1 - distance) * 0.1)

        if self.ball_vel_x > 0 and ai_action == 2:
            reward = np.float32(reward - 0.05)

        return self._get_state(), reward, self.done, {}

    def render(self, screen):
        screen.fill((0, 0, 0))  # Latar belakang hitam

        # Garis putus-putus di tengah
        dash_height = 10
        gap = 10
        for y in range(0, self.height, dash_height + gap):
            pygame.draw.rect(screen, (100, 100, 100), (self.width // 2 - 1, y, 2, dash_height))

        # Paddle kiri (Player)
        pygame.draw.rect(screen, (255, 255, 255), (0, self.player_y, self.paddle_width, self.paddle_height))
        # Paddle kanan (AI)
        pygame.draw.rect(screen, (255, 255, 255), (self.width - self.paddle_width, self.ai_y, self.paddle_width, self.paddle_height))
        # Bola
        pygame.draw.rect(screen, (255, 255, 255), (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()