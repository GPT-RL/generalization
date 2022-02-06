import math
import sys
import time

import pyglet
from my.env import Args, Env, get_meshes
from pyglet.window import key


class ManualControlArgs(Args):
    domain_rand: bool = False
    no_time_limit: bool = False
    top_view: bool = False


if __name__ == "__main__":

    args: ManualControlArgs = ManualControlArgs().parse_args()

    meshes = get_meshes(
        data_path=args.data_path,
        names=args.names,
        obj_pattern=args.obj_pattern,
        png_pattern=args.png_pattern,
    )

    env = Env(
        meshes=meshes,
        size=args.room_size,
    )
    if args.no_time_limit:
        env.max_episode_steps = math.inf
    if args.domain_rand:
        env.domain_rand = True

    view_mode = "top" if args.top_view else "agent"

    env.reset()

    # Create the display window
    env.render("pyglet", view=view_mode)

    def step(action):
        print(
            "step {}/{}: {}".format(
                env.step_count + 1, env.max_episode_steps, env.actions(action).name
            )
        )

        obs, reward, done, info = env.step(action)

        if reward > 0:
            print("reward={:.2f}".format(reward))

        if done:
            print("done!")
            tick = time.time()
            env.reset()
            print(time.time() - tick)

        env.render("pyglet", view=view_mode)

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET")
            env.reset()
            env.render("pyglet", view=view_mode)
            return

        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)

        if symbol == key.UP:
            step(env.actions.move_forward)
        elif symbol == key.DOWN:
            step(env.actions.move_back)

        elif symbol == key.LEFT:
            step(env.actions.turn_left)
        elif symbol == key.RIGHT:
            step(env.actions.turn_right)

        elif symbol == key.PAGEUP or symbol == key.P:
            step(env.actions.pickup)
        elif symbol == key.PAGEDOWN or symbol == key.D:
            step(env.actions.drop)

        elif symbol == key.ENTER:
            step(env.actions.done)

    @env.unwrapped.window.event
    def on_key_release(symbol, modifiers):
        pass

    @env.unwrapped.window.event
    def on_draw():
        env.render("pyglet", view=view_mode)

    @env.unwrapped.window.event
    def on_close():
        pyglet.app.exit()

    # Enter main event loop
    pyglet.app.run()

    env.close()
