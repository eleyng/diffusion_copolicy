# gym-table/gym_table/__init__.py

from gym.envs.registration import register

register(
    id='table-v0',
    entry_point='cooperative_transport.gym_table.envs:TableEnv',
)
