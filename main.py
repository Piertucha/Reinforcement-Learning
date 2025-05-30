import gym
import numpy as np
from gym.envs import register
from gym.wrappers import TimeLimit  # Importujemy wrapper TimeLimit
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
# import stable_baselines3.common.logger, configure  # Jeśli nie używasz, można usunąć

# Parametry, które chcesz przekazać do środowiska
env_kwargs = {
    'L': 0.4,
    'Lh': 0.3,
    'start_pos': (-5, -5),
    'target_pos': (0, 0)
}

# Rejestracja środowiska
register(
    id='MobileRobotTrailerEnv-v0',
    entry_point='my_environment:MobileRobotTrailerEnv',
)

# Definiujemy funkcję tworzącą środowisko z nałożonym wrapperem TimeLimit
def make_env():
    env = gym.make('MobileRobotTrailerEnv-v0', **env_kwargs)
    max_episode_steps = 1000  # Możesz dostosować limit kroków w epizodzie
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

# Przygotowanie środowiska wektorowego z pojedynczym środowiskiem (n_envs=1)
vec_env = make_vec_env(make_env, n_envs=1)

# Parametry dla szumu akcji, które mogą pomóc w eksploracji
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Tworzenie modelu agenta DDPG (ustawienie device='cuda' dla korzystania z GPU)
model = DDPG(
    "MlpPolicy",
    vec_env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./ddpg_robot_trailer_tensorboard/",
    device='cuda'
)

# Callback do oceny i zapisu najlepszego modelu
eval_callback = EvalCallback(
    vec_env,
    best_model_save_path='./ddpg_robot_trailer_best_model/',
    log_path='./ddpg_robot_trailer_results/',
    eval_freq=500,
    deterministic=True,
    render=False  # Renderowanie wyłączone w ewaluacji
)

# Łączymy callbacki w jedną listę – teraz pozostaje tylko EvalCallback
callbacks = CallbackList([eval_callback])

# Nauka agenta
total_timesteps = 15000000
model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callbacks)

# Zapisanie nauczonego modelu
model.save("ddpg_robot_trailer")

# Wczytanie nauczonego modelu
model = DDPG.load("ddpg_robot_trailer")

'''
# Opcjonalnie: Testowanie nauczonego modelu – tutaj możesz mieć render, jeśli chcesz
env = make_env()
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()  # Przy testowaniu możesz włączyć render
    if done or truncated:
        obs, _ = env.reset()
env.close()
'''
