import numpy as np
from matplotlib import pyplot as plt
import torch

def showsal(sal, img, caption="",  alpha=0.5, pos=True):

    noticks = (lambda: plt.xticks([]) and plt.yticks([]))
    slots = 3
    if pos:
        sal = sal * (sal >= 0)
    mask = (sal - sal.min()) / (sal.max()-sal.min())

    plt.subplot(1, slots, 1)
    plt.imshow(img)
    noticks()

    plt.subplot(1, slots, 2)
    plt.imshow(img)    
    plt.imshow(sal, cmap='jet', alpha=alpha)  # Set alpha for transparency
    noticks()
    
    plt.subplot(1, slots, 3)
    masked_img = (mask.unsqueeze(-1).numpy() *img).astype(int)
    plt.imshow(masked_img)
    noticks()
