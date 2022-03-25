#! /usr/bin/env python

import habitat
import numpy as np
import pygame
from my.env import Env

if __name__ == "__main__":
    config = habitat.get_config("objectnav_mp3d.yaml")
    env = Env(config=(config))
    env.seed(0)
    s = env.reset()
    highlight_objective = True
    print(env.objective)

    def get_image(s):
        depth = s["depth"].copy()
        rgb = s["rgb"].copy()
        semantic = s["semantic"].copy()
        if highlight_objective:
            objective_ids = env.object_to_ids[env.objective]
            objective_ids = np.array(objective_ids).reshape(-1, 1, 1)
            expanded = np.expand_dims(semantic, 0)
            is_objective = expanded == objective_ids
            is_objective = is_objective.any(0)
            in_range = depth.squeeze(-1) == 0  # <= config.TASK.SUCCESS_DISTANCE
            highlight = is_objective & in_range
            rgb[:, :, 0][highlight] = 0
            depth[:, :, 0][is_objective] = 0
        semantic = np.expand_dims(semantic, 2)
        depth, rgb, semantic = np.broadcast_arrays(depth, rgb, semantic)

        image = np.concatenate([depth * 255, rgb, semantic], axis=1)
        return image.swapaxes(0, 1)

    shape = get_image(s).shape[:2]
    pygame.init()
    screen = pygame.display.set_mode(shape)

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = "STOP"
                if event.key == pygame.K_UP:
                    action = "MOVE_FORWARD"
                if event.key == pygame.K_LEFT:
                    action = "TURN_LEFT"
                if event.key == pygame.K_RIGHT:
                    action = "TURN_RIGHT"
                if event.key == pygame.K_PAGEUP:
                    action = "LOOK_UP"
                if event.key == pygame.K_PAGEDOWN:
                    action = "LOOK_DOWN"
                if event.key == pygame.K_COMMA:
                    highlight_objective = not highlight_objective
                if event.key == pygame.K_PERIOD:
                    breakpoint()

        obs_array = get_image(s)
        surface = pygame.surfarray.make_surface(obs_array)
        screen.blit(surface, (0, 0))
        pygame.display.update()

        if action is not None:
            s, r, t, i = env.step(action)
            print(i)
            if env.get_done(s):
                print("Episode finished")
                s = env.reset()
