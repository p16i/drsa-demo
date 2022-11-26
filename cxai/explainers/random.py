import numpy as np
import torch


from .base import Explainer


class RandomExplainer(Explainer):
    def __init__(self, model, seed=1, **kwargs):
        np.random.seed(seed)

        self.seed = seed
        self.model = model

    @torch.no_grad()
    def explain(self, x: torch.tensor, label: int) -> (np.array, np.array):
        assert len(x.shape) == 3

        # we simply return a explanation constructed from gaussian noise
        logits = self.model(x.unsqueeze(0)).squeeze().detach().cpu().numpy()

        heatmap = np.random.randn(*x.shape)

        assert len(logits.shape) == 1
        assert len(heatmap.shape) == 3 and heatmap.shape == x.shape

        return logits, heatmap

    def __str__(self):
        return f"random-seed-{self.seed}"


__all__ = ["RandomExplainer"]
