# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import time
import tqdm
import glob
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy

#----------------------------------------------------------------------------

class Renderer(object):

    def __init__(self, generator, discriminator=None, program=None):
        self.generator = generator
        self.discriminator = discriminator
        self.sample_tmp = 0.65
        self.program = program
        self.seed = 0

        if (program is not None) and (len(program.split(':')) == 2):
            from training.dataset import ImageFolderDataset
            self.image_data = ImageFolderDataset(program.split(':')[1])
            self.program = program.split(':')[0]
        else:
            self.image_data = None

    def set_random_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def __call__(self, *args, **kwargs):
        self.generator.eval()  # eval mode...

        if self.program is None:
            if hasattr(self.generator, 'get_final_output'):
                return self.generator.get_final_output(*args, **kwargs)
            return self.generator(*args, **kwargs)
        
        if self.image_data is not None:
            batch_size = 1
            indices = (np.random.rand(batch_size) * len(self.image_data)).tolist()
            rimages = np.stack([self.image_data._load_raw_image(int(i)) for i in indices], 0)
            rimages = torch.from_numpy(rimages).float().to(kwargs['z'].device) / 127.5 - 1
            kwargs['img'] = rimages
        
        outputs = getattr(self, f"render_{self.program}")(*args, **kwargs)
        
        if self.image_data is not None:
            imgs = outputs if not isinstance(outputs, tuple) else outputs[0]
            size = imgs[0].size(-1)
            rimg = F.interpolate(rimages, (size, size), mode='bicubic', align_corners=False)
            imgs = [torch.cat([img, rimg], 0) for img in imgs]
            outputs = imgs if not isinstance(outputs, tuple) else (imgs, outputs[1])
        return outputs

    def get_additional_params(self, ws, t=0):
        gen = self.generator.synthesis
        batch_size = ws.size(0)

        kwargs = {}
        if not hasattr(gen, 'get_latent_codes'):
            return kwargs

        s_val, t_val, r_val = [[0, 0, 0]], [[0.5, 0.5, 0.5]], [0.]
        # kwargs["transformations"] = gen.get_transformations(batch_size=batch_size, mode=[s_val, t_val, r_val], device=ws.device)
        # kwargs["bg_rotation"] = gen.get_bg_rotation(batch_size, device=ws.device)
        # kwargs["light_dir"] = gen.get_light_dir(batch_size, device=ws.device)
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
        kwargs["camera_matrices"] = self.get_camera_traj(t, ws.size(0), device=ws.device)
        return kwargs

    def get_camera_traj(self, t, batch_size=1, traj_type='pigan', device='cpu'):
        gen = self.generator.synthesis
        if traj_type == 'pigan':
            range_u, range_v = gen.C.range_u, gen.C.range_v
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi/2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
            cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device)
        else:
            raise NotImplementedError
        return cam
   
    def render_rotation_camera(self, *args, **kwargs):
        batch_size, n_steps = 2, kwargs["n_steps"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        # ws = ws.repeat(batch_size, 1, 1)

        # kwargs["not_render_background"] = True
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relatve_range_u = kwargs['relative_range_u']
        u_samples = np.linspace(relatve_range_u[0], relatve_range_u[1], n_steps)
        for step in tqdm.tqdm(range(n_steps)):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[u, 0.5, 0.5], device=ws.device)
            cameras.append(gen.get_camera(batch_size=batch_size, mode=[u, 0.5, 0.5], device=ws.device))
            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)

        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out

    def render_rotation_camera3(self, styles=None, *args, **kwargs): 
        gen = self.generator.synthesis
        n_steps = 36  # 120

        if styles is None:
            batch_size = 2
            if 'img' not in kwargs:
                ws = self.generator.mapping(*args, **kwargs)
            else:
                ws = self.generator.encoder(kwargs['img'])['ws']
            # ws = ws.repeat(batch_size, 1, 1)
        else:
            ws = styles
            batch_size = ws.size(0)

        # kwargs["not_render_background"] = True
        # Get Random codes and bg rotation
        self.sample_tmp = 0.72
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        kwargs['noise_mode'] = 'const'
        
        out = []
        tspace = np.linspace(0, 1, n_steps)
        range_u, range_v = gen.C.range_u, gen.C.range_v
        
        for step in tqdm.tqdm(range(n_steps)):
            t = tspace[step]
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi/2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
            
            kwargs["camera_matrices"] = gen.get_camera(
                batch_size=batch_size, mode=[u, v, t], device=ws.device)
            
            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img'] 
            out.append(out_i)
        return out

    def render_rotation_both(self, *args, **kwargs): 
        gen = self.generator.synthesis
        batch_size, n_steps = 1, 36 
        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        ws = ws.repeat(batch_size, 1, 1)

        # kwargs["not_render_background"] = True
        # Get Random codes and bg rotation
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
        kwargs.pop('img', None) 

        out = []
        tspace = np.linspace(0, 1, n_steps)
        range_u, range_v = gen.C.range_u, gen.C.range_v
        
        for step in tqdm.tqdm(range(n_steps)):
            t = tspace[step]
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi/2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])

            kwargs["camera_matrices"] = gen.get_camera(
                batch_size=batch_size, mode=[u, v, 0.5], device=ws.device)
            
            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']  

                kwargs_n = copy.deepcopy(kwargs)
                kwargs_n.update({'render_option': 'early,no_background,up64,depth,normal'})               
                out_n = gen(ws, **kwargs_n)
                out_n = F.interpolate(out_n, 
                    size=(out_i.size(-1), out_i.size(-1)), 
                    mode='bicubic', align_corners=True)
                out_i = torch.cat([out_i, out_n], 0)
            out.append(out_i)
        return out

    def render_rotation_grid(self, styles=None, return_cameras=False, *args, **kwargs):
        gen = self.generator.synthesis
        if styles is None:
            batch_size = 1
            ws = self.generator.mapping(*args, **kwargs)
            ws = ws.repeat(batch_size, 1, 1)
        else:
            ws = styles
            batch_size = ws.size(0)

        kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
        kwargs.pop('img', None) 

        if getattr(gen, "use_voxel_noise", False):
            kwargs['voxel_noise'] = gen.get_voxel_field(styles=ws, n_vols=128, return_noise=True)

        out = []
        cameras = []
        range_u, range_v = gen.C.range_u, gen.C.range_v

        a_steps, b_steps = 6, 3
        aspace = np.linspace(-0.4, 0.4, a_steps)
        bspace = np.linspace(-0.2, 0.2, b_steps) * -1
        for b in tqdm.tqdm(range(b_steps)):
            for a in range(a_steps):
                t_a = aspace[a]
                t_b = bspace[b]
                camera_mat = gen.camera_matrix.repeat(batch_size, 1, 1).to(ws.device)
                loc_x = np.cos(t_b) * np.cos(t_a)
                loc_y = np.cos(t_b) * np.sin(t_a)
                loc_z = np.sin(t_b)
                loc = torch.tensor([[loc_x, loc_y, loc_z]], dtype=torch.float32).to(ws.device)
                from dnnlib.camera import look_at
                R = look_at(loc)
                RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
                RT[:, :3, :3] = R
                RT[:, :3, -1] = loc

                world_mat = RT.to(ws.device)
                #kwargs["camera_matrices"] = gen.get_camera(
                #     batch_size=batch_size, mode=[u, v, 0.5], device=ws.device)
                kwargs["camera_matrices"] = (camera_mat, world_mat, "random", None)

                with torch.no_grad():
                    out_i = gen(ws, **kwargs)
                    if isinstance(out_i, dict):
                        out_i = out_i['img']

                    # kwargs_n = copy.deepcopy(kwargs)
                    # kwargs_n.update({'render_option': 'early,no_background,up64,depth,normal'})
                    # out_n = gen(ws, **kwargs_n)
                    # out_n = F.interpolate(out_n,
                    #                       size=(out_i.size(-1), out_i.size(-1)),
                    #                       mode='bicubic', align_corners=True)
                    # out_i = torch.cat([out_i, out_n], 0)
                out.append(out_i)

        if return_cameras:
            return out, cameras
        else:
            return out

    def render_rotation_camera_grid(self, *args, **kwargs): 
        batch_size, n_steps = 1, 60
        gen = self.generator.synthesis
        bbox_generator = self.generator.synthesis.boundingbox_generator
        
        ws = self.generator.mapping(*args, **kwargs)
        ws = ws.repeat(batch_size, 1, 1)

        # Get Random codes and bg rotation
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
        del kwargs['render_option']

        out = []
        for v in [0.15, 0.5, 1.05]:
            for step in tqdm.tqdm(range(n_steps)):
                # Set Camera
                u = step * 1.0 / (n_steps - 1) - 1.0 
                kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=ws.device)
                with torch.no_grad():
                    out_i = gen(ws, render_option=None, **kwargs)
                    if isinstance(out_i, dict):
                        out_i = out_i['img']
                    # option_n = 'early,no_background,up64,depth,direct_depth'
                    # option_n = 'early,up128,no_background,depth,normal'                
                    # out_n = gen(ws, render_option=option_n, **kwargs)
                    # out_n = F.interpolate(out_n, 
                    #     size=(out_i.size(-1), out_i.size(-1)), 
                    #     mode='bicubic', align_corners=True)
                    # out_i = torch.cat([out_i, out_n], 0)
            
                out.append(out_i)

        # out += out[::-1]
        return out

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
@click.option('--n_steps', default=8, type=int, help="number of steps for each seed")
@click.option('--no-video', default=False)
@click.option('--relative_range_u_scale', default=1.0, type=float, help="relative scale on top of the original range u")
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    render_program=None,
    render_option=None,
    n_steps=8,
    no_video=False,
    relative_range_u_scale=1.0
):

    
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        D = network['D'].to(device)
    # from fairseq import pdb;pdb.set_trace()
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # avoid persistent classes... 
    from training.networks import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)
    G2 = Renderer(G2, D, program=render_program)
    
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i) for i in img]
        all_imgs += [imgs]
    
    else:
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            G2.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(2, G.z_dim)).to(device)
            relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
            outputs = G2(
                z=z,
                c=label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(outputs, tuple):
                img, cameras = outputs
            else:
                img = outputs

            if isinstance(img, List):
                imgs = [proc_img(i) for i in img]
                if not no_video:
                    all_imgs += [imgs]
           
                curr_out_dir = os.path.join(outdir, 'seed_{:0>6d}'.format(seed))
                os.makedirs(curr_out_dir, exist_ok=True)

                if (render_option is not None) and ("gen_ibrnet_metadata" in render_option):
                    intrinsics = []
                    poses = []
                    _, H, W, _ = imgs[0].shape
                    for i, camera in enumerate(cameras):
                        intri, pose, _, _ = camera
                        focal = (H - 1) * 0.5 / intri[0, 0, 0].item()
                        intri = np.diag([focal, focal, 1.0, 1.0]).astype(np.float32)
                        intri[0, 2], intri[1, 2] = (W - 1) * 0.5, (H - 1) * 0.5

                        pose = pose.squeeze().detach().cpu().numpy() @ np.diag([1, -1, -1, 1]).astype(np.float32)
                        intrinsics.append(intri)
                        poses.append(pose)

                    intrinsics = np.stack(intrinsics, axis=0)
                    poses = np.stack(poses, axis=0)

                    np.savez(os.path.join(curr_out_dir, 'cameras.npz'), intrinsics=intrinsics, poses=poses)
                    with open(os.path.join(curr_out_dir, 'meta.conf'), 'w') as f:
                        f.write('depth_range = {}\ntest_hold_out = {}\nheight = {}\nwidth = {}'.
                                format(G2.generator.synthesis.depth_range, 2, H, W))

                img_dir = os.path.join(curr_out_dir, 'images_raw')
                os.makedirs(img_dir, exist_ok=True)
                for step, img in enumerate(imgs):
                    PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB').save(f'{img_dir}/{step:03d}.png')

            else:
                img = proc_img(img)[0]
                PIL.Image.fromarray(img.numpy(), 'RGB').save(f'{outdir}/seed_{seed:0>6d}.png')

    if len(all_imgs) > 0 and (not no_video):
         # write to video
        timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
        seeds = ','.join([str(s) for s in seeds]) if seeds is not None else 'projected'
        network_pkl = network_pkl.split('/')[-1].split('.')[0]
        all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
        imageio.mimwrite(f'{outdir}/{network_pkl}_{timestamp}_{seeds}.mp4', all_imgs, fps=30, quality=8)
        outdir = f'{outdir}/{network_pkl}_{timestamp}_{seeds}'
        os.makedirs(outdir, exist_ok=True)
        for step, img in enumerate(all_imgs):
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
