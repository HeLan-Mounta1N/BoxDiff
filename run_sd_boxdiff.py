
import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import numpy as np
from utils.drawer import draw_rectangle, DashedImageDraw
import imageio
from diffusers import DPMSolverMultistepInverseScheduler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "/home/zideliu/boxdiff/models/zeroscope_v2_576w"
        # If you cannot access the huggingface on your server, you can use the local prepared one.
        # stable_diffusion_version = "../../packages/huggingface/hub/stable-diffusion-v1-4"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version, torch_dtype=torch.float16).to(device)
    # stable.scheduler = DPMSolverMultistepInverseScheduler.from_config(stable.scheduler.config)
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: BoxDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    num_frames=12,
                    bbox=config.bbox,
                    height=512,
                    width=512,
                    config=config)
    frames = outputs.frames
    return frames


@pyrallis.wrap()
def main(config: RunConfig):
    print("bbox shape:", len(config.bbox), '\t', len(config.bbox[0]))
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices
   
    for seed in config.seeds:
        print(f"Current seed is : {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore(height=512,width=512)
    # stable.enable_model_cpu_offload()
        # stable.enable_vae_slicing()
        frames = run_on_prompt(prompt=config.prompt,
                                model=stable,
                                controller=controller,
                                token_indices=token_indices,
                                seed=g,
                                config=config)
        prompt_output_path = config.output_path / config.prompt[:100]
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        canvas_list = []
        for idx in range(len(config.bbox[0])):
            canvas = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8) + 220)
            draw = DashedImageDraw(canvas)
            for i in range(len(config.bbox)):
                x1, y1, x2, y2 = config.bbox[i][idx]
                x2 = x1 + x2
                y2 = y1 + y2
                draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[i], width=5)
            canvas_list.append(np.array(canvas))
            # canvas.save(prompt_output_path / f'{seed}_canvas.png')
        imageio.mimsave(prompt_output_path / f'{seed}.gif',frames, loop=10)
        imageio.mimsave(prompt_output_path / f'{seed}_canvas.gif', canvas_list, loop=10)


if __name__ == '__main__':
    main()
