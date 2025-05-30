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
        self.action_space = spaces.Box(low=np.array([-0.5, -0.1]), high=np.array([0.5, 0.0]),
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

    def reset(self, *, seed=None, options=None):
        # Opcjonalne ustawienie ziarna, jeśli przekazano wartość
        if seed is not None:
            self.seed(seed)
        
        # Inicjalizacja stanu środowiska – możesz dostosować zakresy losowania do swoich potrzeb
        self.state[0] = np.random.uniform(-8, 8)
        self.state[1] = np.random.uniform(-8, 8)
        self.state[2] = np.random.uniform(-np.pi, np.pi)
        self.state[3] = 0  # Ustawienie początkowej orientacji przyczepy

        # Zwracamy krotkę (obserwacja, info) – info może być pustym słownikiem, jeśli nie przekazujemy dodatkowych informacji
        return self.state.astype(np.float32), {}


    def step(self, action):
        u1, u2 = action  # Rozpakowanie akcji
        state = self.state.copy()

        # Ustalanie granic obszaru
        x_min, x_max = -100, 100
        y_min, y_max = -100, 100

        dt = 0.01
        n_steps = 100

        def dynamics(s):
            x, y, theta, beta = s
            dot_theta = u1 * -(self.Lh / self.L) * np.cos(beta) + u2 * (1 / self.L) * np.sin(beta)
            dot_x = u1 * self.Lh * np.sin(beta) * np.cos(theta) + u2 * np.cos(theta) * np.cos(beta)
            dot_y = u1 * self.Lh * np.sin(beta) * np.sin(theta) + u2 * np.sin(theta) * np.cos(beta)
            dot_beta = u1 + u1 * (self.Lh / self.L) * np.cos(beta) - u2 * (1 / self.L) * np.sin(beta)
            return np.array([dot_x, dot_y, dot_theta, dot_beta])
        
        for _ in range(n_steps):
            k1 = dynamics(state)
            k2 = dynamics(state + dt/2 * k1)
            k3 = dynamics(state + dt/2 * k2)
            k4 = dynamics(state + dt * k3)
            state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            # Normalizacja kątów:
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
            state[3] = np.arctan2(np.sin(state[3]), np.cos(state[3]))
        self.state = state

        # Obliczanie odległości do celu
        distance_to_target = np.linalg.norm(self.target_pos - np.array([self.state[0], self.state[1]]))
        reward = -(distance_to_target ** 2)

        done = distance_to_target < 0.1
        if distance_to_target < 0.1:
            reward += 1000

        # Sprawdzenie, czy robot wyszedł poza dozwolony obszar
        if self.state[0] < x_min or self.state[0] > x_max or self.state[1] < y_min or self.state[1] > y_max:
            out_of_bounds = True
            reward -= 10
        else:
            out_of_bounds = False

        # Kara za czas trwania epizodu
        reward -= 0.01

        reward -= 10 * (self.state[3] ** 2)
        action_penalty = 0.1 * (u1**2 + u2**2)
        reward -= action_penalty

        # Dodajemy warunek przerwania epizodu, gdy |beta| przekracza 120° (czyli 2π/3 rad)
        if abs(self.state[3]) > (2 * np.pi / 3):
            done = True
            # Możesz ustawić truncated = True, jeśli chcesz odróżnić przerwanie epizodu od naturalnego zakończenia
            truncated = True
            # Dodatkowa kara za przekroczenie dozwolonego zakresu kąta
            reward -= 30 * abs(self.state[3])
        else:
            truncated = False

        # Ustalanie flag zakończenia
        done = done or out_of_bounds
        terminated = done

        # print("Stan:", self.state)
        return self.state.astype(np.float32), reward, terminated, truncated, {}



    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # Set any other random number generators you might be using
        np.random.seed(seed)
        return [seed]

    def render(self, mode='human'):
        # Ustalona rozdzielczość ekranu
        screen_width = 1920
        screen_height = 1080

        # Jeśli nie ma atrybutu zoom, ustaw domyślnie na 1.0
        if not hasattr(self, "zoom"):
            self.zoom = 10.0

        # Rozmiary obiektów – skalowane przez zoom
        robot_width = int(0.5 * self.zoom)
        robot_height = int(0.3 * self.zoom)
        trailer_width = int(0.4 * self.zoom)
        trailer_height = int(0.2 * self.zoom)
        target_size = int(0.1 * self.zoom)  # Rozmiar reprezentacji celu

        # Inicjalizacja okna, jeśli jeszcze nie utworzono
        if self.viewer is None:
            pygame.init()
            pygame.font.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        # Wypełnienie tła białym kolorem
        self.viewer.fill((255, 255, 255))

        # Wyznaczenie środka ekranu
        center_x, center_y = screen_width / 2, screen_height / 2

        # Rysowanie celu (target) – zielony kwadrat.
        target_x_screen = self.target_pos[0] * self.zoom + center_x
        target_y_screen = self.target_pos[1] * self.zoom + center_y
        target_surface = pygame.Surface((target_size, target_size))
        target_surface.fill((0, 255, 0))
        target_rect = target_surface.get_rect(center=(target_x_screen, target_y_screen))
        self.viewer.blit(target_surface, target_rect.topleft)

        # Pobranie stanu przyczepy (trailer): (x, y) to pozycja przyczepy, theta jej orientacja, beta – kąt między przyczepą a robotem.
        x, y, theta, beta = self.state
        # Przeliczenie współrzędnych na ekran (skalowanie przez zoom)
        x_screen = x * self.zoom
        y_screen = y * self.zoom

        # Rysowanie przyczepy (trailer)
        trailer_center = (x_screen + center_x, y_screen + center_y)
        trailer_surface = pygame.Surface((trailer_width, trailer_height), pygame.SRCALPHA)
        trailer_surface.fill((255, 0, 0))
        rotated_trailer = pygame.transform.rotate(trailer_surface, math.degrees(-theta))
        trailer_rect = rotated_trailer.get_rect(center=trailer_center)
        self.viewer.blit(rotated_trailer, trailer_rect.topleft)

        # Obliczenie punktu zaczepienia (hitch point) na przyczepie.
        # Hitch point znajduje się w odległości L w kierunku przeciwnym do orientacji przyczepy.
        hitch_x = x + self.L * math.cos(theta)
        hitch_y = y + self.L * math.sin(theta)
        hitch_screen = (hitch_x * self.zoom + center_x, hitch_y * self.zoom + center_y)

        # Rysowanie hitch point jako mały czarny okrąg.
        hitch_radius = max(int(0.02 * self.zoom), 0.1)
        pygame.draw.circle(self.viewer, (0, 0, 0), (int(hitch_screen[0]), int(hitch_screen[1])), hitch_radius)

        # Rysowanie linii łączącej trailer (środek) z hitch point.
        pygame.draw.line(self.viewer, (0, 0, 0), trailer_center, hitch_screen, 2)

        # Obliczenie orientacji robota: φ = θ + β
        phi = theta + beta
        # Pozycja robota: od hitch point przesunięte o Lh w kierunku φ.
        robot_x = hitch_x + self.Lh * math.cos(phi)
        robot_y = hitch_y + self.Lh * math.sin(phi)
        robot_center = (robot_x * self.zoom + center_x, robot_y * self.zoom + center_y)

        # Rysowanie robota (niebieski prostokąt) z orientacją φ.
        robot_surface = pygame.Surface((robot_width, robot_height), pygame.SRCALPHA)
        robot_surface.fill((0, 0, 255))
        rotated_robot = pygame.transform.rotate(robot_surface, math.degrees(-phi))
        robot_rect = rotated_robot.get_rect(center=robot_center)
        self.viewer.blit(rotated_robot, robot_rect.topleft)

        # Rysowanie linii łączącej hitch point z robotem.
        pygame.draw.line(self.viewer, (0, 0, 0), hitch_screen, robot_center, 2)

        # --- Dodanie wyświetlania stanów w lewym górnym rogu ---
        font = pygame.font.SysFont("Arial", int(16))
        text = f"x: {x:.2f}  y: {y:.2f}  theta: {math.degrees(theta):.1f}°  beta: {math.degrees(beta):.1f}°"
        text_surface = font.render(text, True, (0, 0, 0))
        self.viewer.blit(text_surface, (10, 10))
        # ----------------------------------------------------

        # Aktualizacja ekranu
        pygame.display.flip()

        # Obsługa zdarzeń – w tym także sterowanie zoomem.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                # Przybliżenie: strzałka w górę
                if event.key == pygame.K_UP:
                    self.zoom *= 1.1
                # Oddalenie: strzałka w dół
                elif event.key == pygame.K_DOWN:
                    self.zoom /= 1.1







'''
# Parametry środowiska
L = 1.0  # odległość między osią koła robota a punktem mocowania przyczepy
Lh = 1.0  # odległość od punktu mocowania przyczepy do punktu kierującego przyczepy
start_pos = (0, 0)  # początkowa pozycja robota
target_pos = (5, 5)  # docelowa pozycja

# Utworzenie instancji środowiska
env = MobileRobotTrailerEnv(L, Lh, start_pos, target_pos)
'''
