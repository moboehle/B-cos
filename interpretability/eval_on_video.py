import os
import shutil
from os.path import join

import torch
import PIL
import numpy as np

import cv2
import matplotlib.pyplot as plt
import imageio
from data.data_transforms import AddInverse
from interpretability.utils import grad_to_img
from project_utils import Str2List, to_numpy


def load_video(path, relative_box=[0., 1., 0., 1.], relative_times=[0., 1.]):

    # Opens the Video file
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    frames = []
    w1, w2, h1, h2 = relative_box
    start, end = relative_times
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        frames.append(frame[int(h1*h):int(h2*h), int(w1*w):int(w2*w)])

    cap.release()
    cv2.destroyAllWindows()
    total_frames = len(frames)

    return frames[int(start*total_frames):int(end*total_frames)], fps


@torch.no_grad()
def most_predicted(model, video, img_transforms):
    predictions = []
    for img in video:
        img = img_transforms(PIL.Image.fromarray(img)).cuda()[None]
        predictions.append((model(AddInverse()(img)))[0].argmax().item())
    c_idcs, counts = np.unique(predictions, return_counts=True)
    c_idx = c_idcs[np.argsort(counts)[-1]]

    print("Most predicted class:", c_idx)
    return c_idx


def process_video(model, img_transforms, video, class_idx=-1):
    if class_idx == -1:
        print("No class index provided, calculating most predicted class.")
        class_idx = most_predicted(model, video, img_transforms=img_transforms)

    atts = []
    imgs = []
    for img in video:
        model.zero_grad()
        img = AddInverse()(img_transforms(PIL.Image.fromarray(img)).cuda()[:][None]).requires_grad_(True)
        out = model(img)[0, class_idx]
        out.backward()
        att = grad_to_img(img[0], img.grad[0], alpha_percentile=100, smooth=5)
        att[..., -1] *= to_numpy(out.sigmoid())
        atts.append(to_numpy(att))
        imgs.append(np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8))

    return imgs, atts


def save_video(imgs, atts, fps, gif_name="my.gif", path="gifs", dpi=75, imsize=224):
    folder = "tmp"
    os.makedirs(join(path, folder), exist_ok=True)
    for idx, att in enumerate(atts):
        fig, ax = plt.subplots(1, figsize=(8, 4))
        plt.imshow(imgs[idx], extent=(0, imsize, 0, imsize))
        plt.imshow(atts[idx], extent=(imsize, 2 * imsize, 0, imsize))
        plt.xlim(0, 2 * imsize)
        plt.xticks([])
        plt.yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig(join(path, folder, "idx_{:03d}.png".format(idx)), bbox_inches='tight', dpi=dpi)
        plt.close()
    images = []
    filenames = [join(path, folder, "idx_{:03d}.png".format(idx)) for idx in range(len(atts))]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(join(path, gif_name), images, fps=fps)
    print(f"GIF saved under {join(path, gif_name)}")
    shutil.rmtree(join(path, folder))
