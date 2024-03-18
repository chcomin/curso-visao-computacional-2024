import matplotlib.pyplot as plt
import torch

def create_image(func, inputs, eps=0.1, grid_resolution=300):
    """Create an image containing the output of a function for each point 
    of a grid. Used for visualizing the decision surface of a model."""

    x0, x1 = inputs.T

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    xx0, xx1 = torch.meshgrid(
        torch.linspace(x0_min, x0_max, grid_resolution),
        torch.linspace(x1_min, x1_max, grid_resolution),
        indexing='xy'
    )

    data_grid = torch.stack((xx0.reshape(-1), xx1.reshape(-1)), dim=1)

    response = func(data_grid)
    response = response.reshape(xx0.shape)

    return response, xx0, xx1

def plot_regions(model, inputs, targets, grid_resolution=300, eps=0.5):
    """Plot the output of a model for a dense grid of points bounded by the data."""

    def get_probs(inputs):

        with torch.no_grad():
            scores = model(inputs)
        return 1/(1+torch.exp(-scores))

    response, xx0, xx1 = create_image(get_probs, inputs, eps=eps, grid_resolution=grid_resolution)

    fig, ax = plt.subplots()
    co = ax.pcolormesh(xx0, xx1, response)
    ax.scatter(*inputs[targets==0].T, s=3, c='C0')
    ax.scatter(*inputs[targets==1].T, s=3, c='C1')
    cbar = fig.colorbar(co, ax=ax, label='P(C=1|X)')
