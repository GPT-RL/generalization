import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pyglet  # noqa: E402

pyglet.options["headless"] = True
pyglet.options["headless_device"] = 1

window = pyglet.window.Window()
label = pyglet.text.Label(
    "Hello, world!",
    font_size=36,
    x=window.width // 2,
    y=window.height // 2,
    anchor_x="center",
    anchor_y="center",
)


@window.event
def on_draw():
    window.clear()
    label.draw()


pyglet.app.run()
