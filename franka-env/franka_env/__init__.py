from gym.envs.registration import register

register(
    id="Franka-v1",
    entry_point="franka_env.envs:FrankaEnv",
    max_episode_steps=400,
)
