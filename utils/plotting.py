import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_comparison(comparisons: List[List[np.array]]):
    cols = len(comparisons[0])
    assert cols == 4 or cols == 3,\
         f'Each comparison must contain 4 images (rgb, depth_gt, depth, b) or 3 images (rgb, depth, b)'

    if cols == 4:
        plot_rgb_depth_prediction_uncertainty(comparisons)
    else:
        plot_rgb_prediction_uncertainty(comparisons)

    

def plot_rgb_depth_prediction_uncertainty(comparisons):
    cols = len(comparisons[0])
    rows = len(comparisons)

    fig, ax = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            if rows == 1:
                ax[j].xaxis.set_visible(False)
                ax[j].yaxis.set_visible(False)
                ax[j].set_aspect('equal')
                
                if j == 0:
                    ax[j].imshow(comparisons[i][j])
                    ax[j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 1:
                    ax[j].imshow(comparisons[i][j], cmap='hot')
                    ax[j].text(0.5,-0.2, "Ground truth", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 2:
                    ax[j].imshow(comparisons[i][j], cmap='hot')
                    ax[j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 3:
                    ax[j].imshow(comparisons[i][j], cmap='inferno')
                    ax[j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[j].transAxes)   
            else:
                if j == 0:
                    ax[i][j].imshow(comparisons[i][j])
                elif j == 3:
                    ax[i][j].imshow(comparisons[i][j], cmap='inferno')
                else:
                    ax[i][j].imshow(comparisons[i][j], cmap='hot')

                ax[i][j].xaxis.set_visible(False)
                ax[i][j].yaxis.set_visible(False)
                ax[i][j].set_aspect('equal')

                if i == rows - 1:
                    if j == 0:
                        ax[i][j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 1:
                        ax[i][j].text(0.5,-0.2, "Ground truth", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 2:
                        ax[i][j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 3:
                        ax[i][j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[i][j].transAxes)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_rgb_prediction_uncertainty(comparisons):
    cols = len(comparisons[0])
    rows = len(comparisons)

    fig, ax = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            if rows == 1:
                ax[j].xaxis.set_visible(False)
                ax[j].yaxis.set_visible(False)
                ax[j].set_aspect('equal')
                
                if j == 0:
                    ax[j].imshow(comparisons[i][j])
                    ax[j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 1:
                    ax[j].imshow(comparisons[i][j], cmap='hot')
                    ax[j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 2:
                    ax[j].imshow(comparisons[i][j], cmap='inferno')
                    ax[j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[j].transAxes)       
            else:
                ax[i][j].xaxis.set_visible(False)
                ax[i][j].yaxis.set_visible(False)
                ax[i][j].set_aspect('equal')

                if j == 0:
                    ax[i][j].imshow(comparisons[i][j])
                elif j == 1:
                    ax[i][j].imshow(comparisons[i][j], cmap='hot')
                elif j == 2:
                    ax[i][j].imshow(comparisons[i][j], cmap='inferno')  

                if i == rows - 1:
                    if j == 0:
                        ax[i][j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 1:
                        ax[i][j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 2:
                        ax[i][j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[i][j].transAxes)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_rgb(image: np.array):
    plt.imshow(image)
    plt.show()

def plot_depth(image: np.array):
    plt.imshow(image, cmap='plasma')
    plt.axis('off')
    

def plot_uncertainty(image: np.array):
    plt.imshow(image, cmap='plasma')
    plt.axis('off')
    