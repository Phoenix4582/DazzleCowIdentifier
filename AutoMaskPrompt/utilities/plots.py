import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import openTSNE
import seaborn as sns
import umap.umap_ as umap

# Define our own plot function
def scatter(x, labels=None, title=None, subtitle=None, filename=None):
    # Get the number of classes (number of unique labels)
    if labels is not None:
        num_classes = np.unique(labels).shape[0]

        # Choose a color palette with seaborn.
        # HLS, MAGMA, ROCKET, MAKO, VIRIDIS, CUBEHELIX
        palette = np.array(sns.color_palette("hls", num_classes+1))
        # palette = np.array(sns.color_palette("hls", 300))

        # Map the colours to different labels
        label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])
        # label_colours = np.array([palette[i] for i in range(labels.shape[0])])

    else:
        # Black dot
        label_colours = 'k'

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect='equal')

    # Set title of image if have one
    if title is not None:
        plt.suptitle(title, size=12, weight='bold')
    if subtitle is not None:
        plt.title(subtitle, size=8, weight='light')

    # Plot the points
    plt.scatter(    x[:,0], x[:,1],
                lw=0, s=10,
                c=label_colours,
                marker="o")

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()

    # Save it to file
    if filename is not None:
        # # print(filename)
        filename_parent = "/".join(filename.split("/")[:-1])
        if not os.path.exists(filename_parent):
            os.makedirs(filename_parent)
        plt.savefig(filename)

    plt.close()

def init_tsne(n_components=2, perplexity=25):
    """
    Initialise a TSNE model with given parameters from OpenTSNE library.
    """
    tsne = openTSNE.TSNE(n_components=n_components,
                         perplexity=perplexity,
                         metric="cosine",
                         n_jobs=8,
                         random_state=42,
                         verbose=False,)

    return tsne

def init_umap(init=None, n_components=2, n_neighbors=25):
    """
    Initialise a UMAP object from UMAP library.
    """
    ump = umap.UMAP(init=init if init is not None else "spectral",
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    metric="cosine",
                    verbose=False,)

    return ump

def create_gif(source:str, dest_folder:str, name:str, format:str = "png", delay:int = 5):
    """
    Create a gif from a folder of images with delay at start and end.
    """
    images = [Image.open(os.path.join(source,im)) for im in os.listdir(source) if im.endswith("."+format)]
    images_start_delay = [images[0] for _ in range(delay)]
    images_end_delay = [images[-1] for _ in range(delay)]
    images = images_start_delay + images + images_end_delay

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    gif_name = name + ".gif"
    images[0].save(os.path.join(dest_folder, gif_name), save_all=True, append_images=images[1:], optimize=False, disposal=2, loop=0, duration=600)
