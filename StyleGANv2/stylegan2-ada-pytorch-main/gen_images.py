# -*- coding: utf-8 -*-


import os
import click
import json
import tempfile
import copy
import torch
import dnnlib
import numpy as np

from tqdm import tqdm

import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True
              , default="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl") 
              #default="./~/training-runs/00003-tfL_celebA_20_point5-auto1-noaug/network-snapshot-001000.pkl")
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH'
              ,default="./~/training-runs/datasets/celebA_20.zip")
@click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)


def calc_metrics(ctx, network_pkl, data,metrics, mirror, gpus, verbose):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    # torch.multiprocessing.set_start_method('spawn')
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     if args.num_gpus == 1:
    #         subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
    #     else:
    #         torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
    
    #Init
    batch_gen=100
    device = torch.device('cuda' if gpus>=1 else 'cpu')
    
    def run_generator(z, c):
        img = G(z=z, c=c)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img
    
    def gen_outputPath(network_pkl,filename):
        path=network_pkl.split("/")
        newpath=""
        for i in range(len(path)-1):
            newpath+=path[i]+"/"
        newpath+=filename
        return newpath
    
    imagesLen=30000
    batchSize=imagesLen/batch_gen
    images = []
    for i in tqdm(range(int(batchSize))):
        G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
        z = torch.randn([batch_gen, G.z_dim], device=device)
        c = [0 for _i in range(batch_gen)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(device)
        image=run_generator(z, c)
        images.append(image.cpu())
    
    images=torch.vstack(images)
    try:
        newpath=gen_outputPath(network_pkl,"generated_images_30k.pt")
        torch.save(images,newpath)
    except:
        torch.save(images,"./datadump/generated_images_30k.pt")
        np.savez("./datadump/generated_images_30k.npz",x=images.numpy())
    
    print ("Saving Generated Images...") 
    return images

if __name__ == "__main__":
    images=calc_metrics() # pylint: disable=no-value-for-parameter
