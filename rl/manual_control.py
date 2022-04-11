#! /usr/bin/env python

import habitat
import my.env
import numpy as np
from my.env import Env


class Args(my.env.Args):
    size: int = 215
    seed: int = 0
    use_attributes: bool = False
    use_pygame: bool = False


def main(*args, seed: int, use_attributes: bool, use_pygame: bool, **kwargs):
    kwargs.update(config=habitat.get_config("objectnav_mp3d.yaml"))
    env = Env(*args, attributes=None, **kwargs)
    env.seed(seed)
    s = env.reset()
    if not use_pygame:
        env.render(mode="ascii", pause=False)
    print(env.obj_id, env.objective)

    def get_image(s):
        rgb = s["rgb"].copy()
        overlay = env.objective_overlay(s)
        overlay = 255 * np.expand_dims(overlay, -1)
        rgb, overlay = np.broadcast_arrays(rgb, overlay)

        image = np.concatenate([rgb, overlay], axis=1)
        return image.swapaxes(0, 1)

    if use_pygame:
        import pygame

        shape = get_image(s).shape[:2]
        pygame.init()
        screen = pygame.display.set_mode(shape)

    while True:
        if use_pygame:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = "STOP"
                    if event.key == pygame.K_UP:
                        action = "MOVE_FORWARD"
                    if event.key == pygame.K_RCTRL:
                        action = "STRAFE_LEFT"
                    if event.key == pygame.K_KP0:
                        action = "STRAFE_RIGHT"
                    if event.key == pygame.K_LEFT:
                        action = "TURN_LEFT"
                    if event.key == pygame.K_RIGHT:
                        action = "TURN_RIGHT"
                    if event.key == pygame.K_PAGEUP:
                        action = "LOOK_UP"
                    if event.key == pygame.K_PAGEDOWN:
                        action = "LOOK_DOWN"
                    if event.key == pygame.K_PERIOD:
                        breakpoint()

            obs_array = get_image(s)
            surface = pygame.surfarray.make_surface(obs_array)
            screen.blit(surface, (0, 0))
            pygame.display.update()

        else:
            action = input("Action: ")
            if action == " ":
                action = "STOP"
            elif action == "w":
                action = "MOVE_FORWARD"
            elif action == "a":
                action = "TURN_LEFT"
            elif action == "d":
                action = "TURN_RIGHT"
            elif action == "q":
                action = "STRAFE_LEFT"
            elif action == "e":
                action = "STRAFE_RIGHT"
            elif action == "e":
                action = "LOOK_UP"
            elif action == "f":
                action = "LOOK_DOWN"
            else:
                action = None

        if action is not None:
            s, r, t, i = env.step(action)
            if not use_pygame:
                env.render(mode="ascii", pause=False)
            if env.get_done(s):
                print("Episode finished")
                s = env.reset()
                print(env.obj_id, env.objective)


if __name__ == "__main__":
    main(**Args().parse_args().as_dict())
