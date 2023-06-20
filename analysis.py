import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.stats import pearsonr

def main():
    environments = ["volcano", "lake", "random", "river",]
    results_directory = r"./results"

    for env in environments:

        df = pd.read_csv(os.path.join(results_directory, "output_"+env+".csv"))

        df_last_gen = df[df.Gen == 50]

        idx = df_last_gen.groupby("Obs")['fitness'].transform(max) == df_last_gen['fitness']

        results_df = df_last_gen[idx]


        pca = PCA(n_components=2)
        x_embedded = pca.fit_transform(results_df[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']].to_numpy())

        print(f"\n{env}\n :party poppers:")
        for i in ['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']:
            for j in ['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']:
                if i == j:
                    break
                print(f"Pearsons correlation between {i} & {j} {pearsonr(results_df[i], results_df[j])}")

    # print(pca.explained_variance_ratio_)
    # plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
    # plt.show()

    # sns.pairplot(df_last_gen[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire', 'fitness']],)
        # sns.pairplot(results_df[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire', 'fitness']], )
        # plt.show()
    
    
        df_added = df_last_gen[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire', 'fitness']].copy()
        df_added["best"] = np.where(idx, 'BEST', 'OTHER')
        sns.pairplot(df_added, hue="best")
        plt.show()

if __name__ == '__main__':
    main()