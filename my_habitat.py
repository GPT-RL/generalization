import itertools

import habitat

from rl.my.env import Env

if __name__ == "__main__":
    env = Env(config=habitat.get_config("objectnav_mp3d.yaml"))
    for i in itertools.count():
        print(i)
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
