import os
from omegaconf import OmegaConf
import cv2
import torch
import torch.nn.functional as F
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.io import read_image
import argparse
import pandas as pd

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict

from ldm.models.diffusion.ddim import DDIMSampler

# for masactrl
from masactrl.masactrl_utils import regiter_attention_editor_ldm
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl import MuraMasa
from masactrl.masactrl import MutualSelfAttentionControlMask
from masactrl.masactrl import MutualSelfAttentionControlMaskAuto

torch.set_grad_enabled(False)

supported_cond = [e.name for e in ExtraCondition]
parser = get_base_argument_parser()
parser.add_argument(
    '--which_cond',
    default='sketch',
    type=str,
    required=False,
    choices=supported_cond,
    help='which condition modality you want to test',
)
# [MasaCtrl added] reference cond path
parser.add_argument(
    "--cond_path_src",
    default="./examples/cond/sketch.png",
    type=str,
    help="the condition image path to synthesize the source image",
)
parser.add_argument(
    "--prompt_src",
    type=str,
    default=None,
    help="the prompt to synthesize the source image",
)
parser.add_argument(
    "--src_img_path",
    type=str,
    default=None,
    help="the input real source image path"
)
parser.add_argument(
    "--use_ddim_latents",
    action='store_true',
    default=False,
    help="whether to use ddim intermediate during the synthesis"
)
parser.add_argument(
    "--start_code_path",
    type=str,
    default=None,
    help="the inverted start code path to synthesize the source image",
)
parser.add_argument(
    "--masa_step",
    type=int,
    default=4,
    help="the starting step for MasaCtrl",
)
parser.add_argument(
    "--masa_layer",
    type=int,
    default=10,
    help="the starting layer for MasaCtrl",
)


opt = parser.parse_args()
opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

supported_cond = [e.name for e in ExtraCondition]
parser = get_base_argument_parser()
opt = parser.parse_args()
which_cond = "sketch"
prompt_src= "A bear walking in the forest"
prompt ="A bear standing in the forest"

outdir = f'./outputs/test-{which_cond}'
os.makedirs(outdir, exist_ok=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# cond_path="./examples/in/"
cond_path="./examples/in/"
image_paths = [os.path.join(cond_path, f) for f in os.listdir(cond_path)]

sd_ckpt = './models/[REAL]Realistic_Vision_V3.0.ckpt'
# sd_ckpt = './models/sd_xl_base_1.0.safetensors'

vae_ckpt = None
config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, sd_ckpt, vae_ckpt)
sd_model = model.to(device)
sampler = DDIMSampler(model)

adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
process_cond_module = getattr(api, f'get_cond_{which_cond}')


STEP =  24
LAYER =  10
steps = 50
seed = 42


with torch.inference_mode(), \
        sd_model.ema_scope(), \
        autocast('cuda'):
    # for test_idx, cond_path in enumerate(image_paths):
    cond_model = None
    seed=seed+1919
    for v_idx in range(20):
        seed=seed+1
        seed_everything(seed)
        
        # seed_everything(opt.seed+v_idx+test_idx)
        # base_count = len(os.listdir(outdir)) // 2
        base_count = len(os.listdir(outdir))
        
        nigger = True
        image=[]
        count=0
        for cond_path in image_paths:
            count=count+1
            cond=process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model,nigger,count=count)
            
            # cv2.imwrite(os.path.join(outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))
            adapter_features, append_to_context = get_adapter_feature(cond, adapter)
            
            image.append(adapter_features)
        '''
        cond_path_src = "./examples/in/sketch_src.png"
        cond_path = "./examples/cond/sketch.png"
        
        
        image=[]
        for cond_path in image_paths:
            image.append(cond_path)
        
        cond1 = process_cond_module(opt, image[0], opt.cond_inp_type, cond_model,nigger)
        cond2 = process_cond_module(opt, image[1], opt.cond_inp_type, cond_model,nigger)
        cond3 = process_cond_module(opt, image[2], opt.cond_inp_type, cond_model,nigger)
        cond4 = process_cond_module(opt, image[3], opt.cond_inp_type, cond_model,nigger)
        
        cv2.imwrite(os.path.join(outdir, f'{base_count:05}_{which_cond}_src.png'), tensor2img(cond2))
        
        adapter_features1, append_to_context1 = get_adapter_feature(cond1, adapter)
        adapter_features2, append_to_context2 = get_adapter_feature(cond2, adapter)
        adapter_features3, append_to_context3 = get_adapter_feature(cond3, adapter)
        adapter_features4, append_to_context4 = get_adapter_feature(cond4, adapter)

        adapter_features = [torch.cat([adapter_features1[i], adapter_features2[i],adapter_features3[i],adapter_features4[i]]) for i in range(len(adapter_features1))]
        # adapter_features = [torch.cat([adapter_features1[i], adapter_features2[i]]) for i in range(len(adapter_features1))]

        adapter_features = [torch.cat([feats] * 2) for feats in adapter_features]
        append_to_context = append_to_context1
        '''
        
        afres = [[] for _ in range(len(image[0]))]
        count=0
        for af in image:
            for count in range(len(af)):
                afres[count].append(af[count])
        af_torch=[]
        for i in afres:
            af_torch.append(torch.cat(i))
                
        adapter_features = [torch.cat([feats] * 2) for feats in af_torch]

        # prepare the batch prompts
        #prompts = [prompt_src, prompt] 
        
        u_prompts=["longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"]
        
        prompts1= [
                "1boy, outdoor,blue tshirt, countryside",  # source prompt
                "1boy, outdoor,blue tshirt, countryside, sitting on a bench", # mid 1
                "1boy, outdoor, blue tshirt,countryside, sitting on a bench, surrounded by trees", # mid 2
                "covered with snow, 1boy,blue tshirt, outdoor, countryside, sitting on a bench, surrounded by trees, ",# target prompt
                ]
        
        constant_prompt1= ",back toward camera,sports wear, white tshirt, black shorts, HDR,UHD,8K, Best quality, Masterpiece, Highly detailed,High resolution,High quality"
        seed=seed+0
        
        prompts2 = [
                "1man, ",  # source prompt
                "1man lifting weights", # mid 1
                "1man spinning", # mid 2
                "1man doing bench press, ",# target prompt
                ]
        
        constant_prompt2 = ",sports wear, white tshirt, black shorts, HDR,UHD,8K, Best quality, Masterpiece, Highly detailed,High resolution,High quality"
        
        prompts3 = [
                "farm, ",  # source prompt
                "farm,cows", # mid 1
                "farm,cows,sheep", # mid 2
                "farm,cows,sheep, eating grass",# target prompt
                ]
        
        constant_prompt3 = ", HDR,UHD,8K, Best quality, Masterpiece, Highly detailed,High resolution,High quality"
        
        prompts = [
                "an empty room ,a table at the center,",  # source prompt
                "a room ,a table at the center,a birthday cake on the table,", # mid 1
                "a room ,a table at the center,a birthday cake on the table,a girl stand next to the birthday cake", # mid 2
                "a room ,a table at the center,a birthday cake on the table,a girl stand next to the birthday cake,celebrating with her mother ,",# target prompt
                ]
        
        constant_prompt = ", HDR,UHD,8K, Best quality, Masterpiece, Highly detailed,High resolution,High quality"
        
        
        
        
        
        
        pmt=[]
        for prompt in prompts:
            prompt = prompt + constant_prompt
            pmt.append(prompt)
            
        prompts = pmt
                    
        '''
        prompts = [
                "1boy, outdoor",  # source prompt
                "1boy, outdoor, sitting on a bench"
                ]
        '''
        '''
        prompts = [
                "A bear standing in the forest",  # source prompt
                "A bear walking in the forest", # mid 1
                ]
        '''
            
        print("promts: ", prompts)
        # get text embedding
        c = sd_model.get_learned_conditioning(prompts)
        if opt.scale != 1.0:
            # uc = sd_model.get_learned_conditioning([""] * len(prompts))
            uc = sd_model.get_learned_conditioning( u_prompts* len(prompts))
        else:
            uc = None
        c, uc = fix_cond_shapes(sd_model, c, uc)

        if not hasattr(opt, 'H'):
            opt.H = 512
            opt.W = 512
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        ref_latents = None
                    
        start_code = torch.randn([1, *shape], device=opt.device)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        # hijack the attention module
        editor = MuraMasa(STEP, LAYER)
        # editor = MutualSelfAttentionControl(STEP, LAYER)
        regiter_attention_editor_ldm(sd_model, editor)

        samples_latents, _ = sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=len(prompts),
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            x_T=start_code,
            features_adapter=adapter_features,
            append_to_context=append_to_context,
            cond_tau=opt.cond_tau,
            style_cond_tau=opt.style_cond_tau,
            ref_latents=ref_latents,
        )

        x_samples = sd_model.decode_first_stage(samples_latents)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        cv2.imwrite(os.path.join(outdir, f'{base_count:05}_all_result.png'), tensor2img(x_samples))
        # save the prompts and seed
        with open(os.path.join(outdir, "log.txt"), "w") as f:
            for prom in prompts:
                f.write(prom)
                f.write("\n")
            f.write(f"seed: {seed}")
        for i in range(len(x_samples)):
            base_count += 1
            cv2.imwrite(os.path.join(outdir, f'{base_count:05}_result.png'), tensor2img(x_samples[i]))
