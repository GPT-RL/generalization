import math
import sys
import time
from inspect import signature
from pathlib import Path

import pyglet
from my.env import Args, Env, Obs
from my.mesh_paths import get_meshes
from PIL import Image
from pyglet.window import key
from transformers import CLIPModel, CLIPProcessor


class ManualControlArgs(Args):
    domain_rand: bool = False
    no_time_limit: bool = False
    top_view: bool = False


if __name__ == "__main__":

    args: ManualControlArgs = ManualControlArgs().parse_args()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    meshes = get_meshes(data_path=Path(args.data_path), names=args.names)
    kwargs = {
        k: v
        for k, v in args.as_dict().items()
        if k in signature(Env.__init__).parameters
    }
    env = Env(meshes=meshes, test=True, **kwargs)
    # env = CLIPProcessorWrapper(env, processor, [m.name for m in meshes])
    # replace = {
    #     "master chef coffee can": "blue, white, beige, cylinder",
    #     "starkist tuna fish can": "blue, silver, red, white, cylinder",
    # }
    # breakpoint()
    # all_missions = [replace[m.name] for m in meshes]
    # tokens = processor.tokenizer(all_missions, return_tensors="pt", padding=True)
    # tokens = tokens["input_ids"]
    # tokens = {m: t for m, t in zip(all_missions, tokens)}
    if args.no_time_limit:
        env.max_episode_steps = math.inf
    if args.domain_rand:
        env.domain_rand = True

    view_mode = "top" if args.top_view else "agent"

    obs = env.reset()

    # Create the display window
    env.render("pyglet", view=view_mode)

    def step(action):
        global obs
        print(f"step {env.step_count + 1}/{env.max_episode_steps}")

        obs, reward, done, info = env.step(action)
        Obs(**obs)

        # text = [m.name for m in env.chosen_meshes]
        # text = [replace[t] for t in text]
        # inputs = processor(
        #     text=text,
        #     images=o.image,
        #     return_tensors="pt",
        #     padding=True,
        # )
        #
        # outputs = model(**inputs)
        # logits_per_image = (
        #     outputs.logits_per_image
        # )  # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1).reshape(
        #     -1
        # )  # we can take the softmax to get the label probabilities
        # for choice, prob in zip(text, probs):
        #     print(f"{choice}: {(100 * prob).round()}%")

        # pixel_values = processor(images=o.image, return_tensors="pt", padding=True)
        #
        # def input_ids():
        #     for m in env.chosen_meshes:
        #         yield tokens[replace[m.name]]
        #
        # input_ids = torch.stack(list(input_ids()), dim=0)
        # assert torch.equal(pixel_values["pixel_values"], inputs["pixel_values"])
        # assert torch.equal(input_ids, inputs["input_ids"])
        # image_embeds = model.vision_model(**pixel_values)[1]
        # image_embeds = model.visual_projection(image_embeds)
        # text_embeds = model.text_model(
        #     input_ids=input_ids, attention_mask=input_ids != 0
        # )[1]
        # text_embeds = model.text_projection(text_embeds)
        #
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # logit_scale = model.logit_scale.exp()
        #
        # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        # logits_per_text = logits_per_text.flatten()
        # probs_per_text = logits_per_text.softmax(0)
        # for m, l, p in zip(env.chosen_meshes, logits_per_text, probs_per_text):
        #     print(m.name, float(l), float(p))

        if reward > 0:
            print(f"reward={reward:.2f}")

        if done:
            print("done!")
            tick = time.time()
            obs = env.reset()
            print(time.time() - tick)
        print(env.mission)

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

        elif symbol == key.SPACE:
            Image.fromarray(Obs(**obs).image).show()

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
