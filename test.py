import gym
import time
from stable_baselines3 import DDPG
from gym.envs import register

env_kwargs = {
    'L': 0.4,
    'Lh': 0.3,
    'start_pos': (10, 10),
    'target_pos': (0, 0)
}

# Rejestrowanie środowiska, jeśli jeszcze nie zostało zarejestrowane
register(
    id='MobileRobotTrailerEnv-v0',
    entry_point='my_environment:MobileRobotTrailerEnv',
    kwargs=env_kwargs
)

# Tworzenie instancji środowiska
env = gym.make('MobileRobotTrailerEnv-v0')
print("Przygotowano środowisko")
# Wczytanie nauczonego modelu
model = DDPG.load("ddpg_robot_trailer")
print("Wczytano model")
# Testowanie nauczonego modelu
episodes = 5  # Liczba epizodów do przetestowania
for episode in range(episodes):
    print("rozpoczęcie pętli")
    obs, _ = env.reset()  # Rozpakowujemy krotkę (obs, info)
    print("env. reset")
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.5)
    print(f"Epizod {episode + 1}: Zakończony")

env.close()
