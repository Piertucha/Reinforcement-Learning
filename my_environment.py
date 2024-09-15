import gym
from gym import spaces
import numpy as np
import pygame
import math
import gym.utils.seeding


class MobileRobotTrailerEnv(gym.Env):
    def __init__(self, L, Lh, start_pos, target_pos):
        super(MobileRobotTrailerEnv, self).__init__()

        # Przestrzeń akcji: u1 - prędkość kątowa, u2 - prędkość postępowa
        self.action_space = spaces.Box(low=np.array([-np.pi * 1 / 90, -0.1]), high=np.array([np.pi * 1 / 90, 0]),
                                       dtype=np.float32)

        # Przestrzeń obserwacji: [x, y, theta, beta] - pozycje i orientacje robota i przyczepy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Parametry robota, przyczepy i środowiska
        self.L = L
        self.Lh = Lh
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)

        # Inicjalizacja atrybutu viewer
        self.viewer = None

        # Stan początkowy
        self.state = np.array([start_pos[0], start_pos[1], 0, 0], dtype=np.float32)
        self.reset()

    def reset(self):
        # print("reset")
        self.state[0] = np.random.uniform(-8, 8)
        self.state[1] = np.random.uniform(-8, 8)
        self.state[2] = np.random.uniform(-np.pi, np.pi)  # Orientacja robota (theta)
        self.state[3] = 0  # Orientacja przyczepy względem robota (beta)
        return self.state.astype(np.float32)

    def step(self, action):
        u1, u2 = action  # prędkość kątowa i postępowa robota
        x, y, theta, beta = self.state

        # Granice obszaru, po którym porusza się robot
        x_min, x_max = -100, 100
        y_min, y_max = -100, 100

        for tp in range(100):
            # Równania kinematyki dla robota i przyczepy
            dot_theta = u1 * -(self.Lh / self.L) * np.cos(beta) + u2 * (1 / self.L) * np.sin(beta)
            dot_x = u1 * self.Lh * np.sin(beta) * np.cos(theta) + u2 * np.cos(theta) * np.cos(beta)
            dot_y = u1 * self.Lh * np.sin(beta) * np.sin(theta) + u2 * np.sin(theta) * np.cos(beta)
            dot_beta = u1 + u1 * (self.Lh / self.L) * np.cos(beta) + u2 * -(1 / self.L) * np.sin(beta)

            # Aktualizacja stanu
            x += dot_x*0.01  # aktualizacja x
            y += dot_y*0.01 # aktualizacja y
            theta += dot_theta*0.01  # aktualizacja theta
            beta += dot_beta*0.01  # aktualizacja beta
            beta = np.arctan2(np.sin(beta), np.cos(beta))
            theta = np.arctan2(np.sin(theta), np.cos(theta))

        self.state = np.array([x, y, theta, beta])
        # print(self.state)
        # Oblicz nagrodę
        distance_to_target = np.linalg.norm(self.target_pos - np.array([x, y]))
        reward = -distance_to_target   # Nagroda jest negatywnie proporcjonalna do odległości od celu
        # możliwa modyfikacja
        # reward = 1/distance_to_target
        done = distance_to_target < 0.1  # Zakończ, gdy robot jest blisko celu
        if distance_to_target < 0.1:
            reward += 1000000

        # Sprawdzenie, czy robot znajduje się w granicach obszaru
        if x < x_min or x > x_max or y < y_min or y > y_max:
            done = True  # Zakończenie epizodu
            reward -= 10  # Kara za wyjście poza obszar
        reward -= 0.01 #Kara za czas trwania epizodu
        if abs(beta) > np.pi / 6:
            reward -= abs(beta)
        if abs(beta) > np.pi / 2:
            reward -= abs(beta) * 2
        if abs(beta) > 2 * np.pi / 3:
            done = True
            reward -= abs(beta) * 30
            # Sprawdź, czy epizod się skończył

        # print('nagroda ' + str(reward) + ' odleglosc: ' + str(distance_to_target))
        # print()
        return self.state.astype(np.float32), reward, done, {}  # Zwróć krotkę

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # Set any other random number generators you might be using
        np.random.seed(seed)
        return [seed]

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        robot_width, robot_height = 30, 50
        trailer_width, trailer_height = 20, 40
        target_size = 10  # Rozmiar reprezentacji celu

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        # Wypełnienie tła
        self.viewer.fill((255, 255, 255))

        center_x, center_y = screen_width / 2, screen_height / 2

        # Rysowanie miejsca docelowego
        target_surface = pygame.Surface((target_size, target_size))
        target_surface.fill((0, 255, 0))
        target_rect = target_surface.get_rect(center=(self.target_pos[0] + center_x, self.target_pos[1] + center_y))
        self.viewer.blit(target_surface, target_rect.topleft)

        # Rysowanie przyczepy
        robot_pos = self.state[0:2]
        robot_angle = self.state[2]
        robot_surface = pygame.Surface((robot_width, robot_height))
        robot_surface.fill((0, 0, 255))
        rotated_robot = pygame.transform.rotate(robot_surface, math.degrees(-robot_angle))
        robot_rect = rotated_robot.get_rect(center=(robot_pos[0] + center_x, robot_pos[1] + center_y))
        self.viewer.blit(rotated_robot, robot_rect.topleft)


        # Aktualizacja ekranu
        pygame.display.flip()
        print(self.state)

        # Sprawdzanie zdarzeń (np. zamknięcia okna)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        pass


'''
# Parametry środowiska
L = 1.0  # odległość między osią koła robota a punktem mocowania przyczepy
Lh = 1.0  # odległość od punktu mocowania przyczepy do punktu kierującego przyczepy
start_pos = (0, 0)  # początkowa pozycja robota
target_pos = (5, 5)  # docelowa pozycja

# Utworzenie instancji środowiska
env = MobileRobotTrailerEnv(L, Lh, start_pos, target_pos)
'''
