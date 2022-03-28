import os
import shutil
import subprocess

import numpy as np
import torch
from torch.hub import download_url_to_file


def savefig(fig, filename, *args, **kwargs):
    """
    Same as fig.savefig except that the results are compressed to make the paper smaller.
    """
    fig.savefig(filename, *args, **kwargs)
    gs_opt(filename)


def gs_opt(filename):
    filename_tmp = filename.split('.')[-2]+'_tmp.pdf'
    gs = ['gs',
          '-sDEVICE=pdfwrite',
          '-dEmbedAllFonts=true',
          '-dSubsetFonts=false',             # Create font subsets (default)
          '-dPDFSETTINGS=/ebook',        # Image resolution
          '-dAutoRotatePages=/None',        # Rotation
          '-dDetectDuplicateImages=true',   # Embeds images used multiple times only once
          '-dCompressFonts=false',           # Compress fonts in the output (default)
          '-dNOPAUSE',                      # No pause after each image
          '-dQUIET',                        # Suppress output
          '-dBATCH',                        # Automatically exit
          '-sOutputFile='+filename_tmp,      # Save to temporary output
          filename]                         # Input file

    subprocess.run(gs)                                      # Create temporary file
#     subprocess.run(['rm', filename])            # Delete input file
    subprocess.run(['mv',filename_tmp,filename]) # Rename temporary to input file


# Downloading precomputed logits for the paper such as neuron activations for plotting in the notebooks.
def get_precomputed_results():
    if os.path.isdir("results"):
        return
    print("No precomputed results found. Necessary for plotting. Downloading now.")
    download_url_to_file("https://nextcloud.mpi-klsb.mpg.de/index.php/s/4H4Xs2a9KyB9Ae3/download", "results.zip", progress=True)
    shutil.unpack_archive("results.zip", extract_dir="results")
    os.remove("results.zip")
    print("Done.")


# Downloading videos for evaluating model on videos (see notebooks).
def get_videos():
    urls = {
        "drake": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/TbX5Yn9qBaHmXrj/download",
        "lorikeet": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/pfnwsbDQTCJcQZL/download",
        "zebra": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/npJxP936yJ95tYE/download"
    }
    for name, url in urls.items():
        if not os.path.isfile(f"resources/{name}.mp4"):
            os.makedirs("resources", exist_ok=True)
            print(f"Video for {name} not found, downloading now.")
            download_url_to_file(url, f"resources/{name}.mp4", progress=True)
    print("Done.")


def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def to_numpy_img(tensor):
    """
    Converting tensor to numpy image. Expects a tensor of at most 3 dimensions in the format (C, H, W),
    which is converted to a numpy array with (H, W, C) or (H, W) if C=1.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    return to_numpy(tensor.permute(1, 2, 0)).squeeze()


def make_exp_name(dataset, base_net, exp_name):
    return "-".join([dataset, base_net, exp_name])


def make_exp_name_from_save_path(save_path):
    dataset, base_net, exp_name = save_path.split("/")[-3:]
    return make_exp_name(dataset, base_net, exp_name)


class Str2List:

    def __init__(self, dtype=int):
        self.dtype = dtype

    def __call__(self, instr):
        return str_to_list(instr, self.dtype)


def str_to_list(s, _type=int):
    """
    Parses a string representation of a list of integers ('[1,2,3]') to a list of integers.
    Used for argument parsers.
    """
    s = s.replace(" ", "")
    assert s.startswith("[") and s.endswith("]"), s
    return np.array(s[1:-1].split(","), dtype=_type).tolist()


def str_to_bool(s):
    """
    Parses a string representation of a list of float values ('[0.1,2,3]') to a list of floats.
    Used for argument parsers.
    """
    return s.lower() in ["yes", "true", "1"]
