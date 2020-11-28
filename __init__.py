from gym.envs.registration import register


register(
        id='network_game-v0',
        entry_point='env.myenv:NetworkGameEnv'
        )
