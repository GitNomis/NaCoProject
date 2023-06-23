import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from sklearn.decomposition import PCA


from scipy.stats import pearsonr


def main():
    environments = [
        "volcano",
        "lake",
        "random",
        "river",
    ]
    results_directory = r"./results"

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    im = None
    for i, env in enumerate(environments):

        df = pd.read_csv(os.path.join(results_directory, "output_" + env + ".csv"))

        df_last_gen = df[df.Gen == 50]

        idx = df_last_gen.groupby("Obs")["fitness"].transform(max) == df_last_gen["fitness"]

        results_df = df_last_gen#[idx]

        X = results_df[["Alignment", "Cohesion", "Separation", "GoToWater", "GoToFire"]].to_numpy()

        print(X.shape)
        pca = PCA(n_components=2)
        x_embedded = pca.fit_transform(X)

        colour = (results_df["fitness"] - results_df["fitness"].min()) / (
            results_df["fitness"].max() - results_df["fitness"].min()
        )

        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["tab:blue", "tab:orange"])
        im = axes[i].scatter(x_embedded[:, 0], x_embedded[:, 1], c=colour.to_numpy())
        axes[i].text(
            0.01,
            0.99,
            f"{np.sum(pca.explained_variance_ratio_):.2f}",
            ha="left",
            va="top",
            transform=axes[i].transAxes,
        )
        axes[i].tick_params(
            axis="y", which="both", left=False,
        )
        axes[i].set_title(env)
    fig.supxlabel("First principal component")
    fig.supylabel("Second principal component")

    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.00)

    cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.7])
    fig.colorbar(im, cax=cb_ax, label="Normalized fitness")
    plt.show()
    fig.savefig("./pca_all.pdf", bbox_inches="tight")