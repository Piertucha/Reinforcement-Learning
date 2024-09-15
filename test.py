import gym
from stable_baselines3 import DDPG
from gym.envs import register
env_kwargs = {
    'L': 1.0,
    'Lh': 1.0,
    'start_pos': (0, 0),
    'target_pos': (5, 5)
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
    obs = env.reset()
    print("env. reset")
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
    print(f"Epizod {episode + 1}: Zakończony")

env.close()