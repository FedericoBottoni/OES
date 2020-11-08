from gym.envs.registration import register

register(
    id='MountainCarCustom-v0',
    entry_point='src.custom_envs:MountainCarCustom',
)