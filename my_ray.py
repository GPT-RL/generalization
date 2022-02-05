import ray

ray.init()


# noinspection PyCallingNonCallable,PyArgumentList
@ray.remote(num_gpus=1)
def f():
    import pyglet

    print(ray.get_gpu_ids())

    pyglet.options["headless"] = True
    from gym_miniworld.envs import OneRoom

    env = OneRoom()
    while True:
        env.reset()
        t = False
        while not t:
            s, r, t, i = env.step(env.action_space.sample())


futures = [f.remote() for i in range(4)]
print(ray.get(futures))  # [0, 1, 4, 9]
