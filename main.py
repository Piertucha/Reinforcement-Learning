import gym
import numpy as np
from gym.envs import register
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger, configure

# Parametry, które chcesz przekazać do środowiska
env_kwargs = {
    'L': 0.4,
    'Lh': 0.1,
    'start_pos': (-5, -5),
    'target_pos': (0, 0)
}

register(
    id='MobileRobotTrailerEnv-v0',
    entry_point='my_environment:MobileRobotTrailerEnv',
)
# Tworzenie instancji środowiska z przekazaniem parametrów
env = gym.make('MobileRobotTrailerEnv-v0', **env_kwargs)

# Stworzenie callbacków
total_timesteps = 1500000

# Przygotowanie środowiska wektorowego dla stabilnej pracy z algorytmami stable_baselines
vec_env = make_vec_env(lambda: env, n_envs=1)

# Parametry dla szumu akcji, które mogą pomóc w eksploracji
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Tworzenie modelu agenta DDPG
model = DDPG(
    "MlpPolicy",
    vec_env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./ddpg_robot_trailer_tensorboard/"
)

# Callback do oceny i zapisu najlepszego modelu dasdasd
eval_callback = EvalCallback(
    vec_env,
    best_model_save_path='./ddpg_robot_trailer_best_model/',
    log_path='./ddpg_robot_trailer_results/',
    eval_freq=500,
    deterministic=True,
    render=False
)

# Nauka agenta
model.learn(total_timesteps=total_timesteps, log_interval=1)
# Zapisanie nauczony modelu
model.save("ddpg_robot_trailer")

# Wczytanie nauczonego modelu
model = DDPG.load("ddpg_robot_trailer")
'''
# Testowanie nauczonego modelu
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()

env.close()
'''