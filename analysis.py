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

        # pca = PCA(n_components=2)
        # x_embedded = pca.fit_transform(results_df[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']].to_numpy())
        # print(pca.explained_variance_ratio_)
        # plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        # plt.show()

        # print(f"\n{env}\n :party poppers:")
        # for i in ['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']:
        #     for j in ['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire']:
        #         if i == j:
        #             break
        #         print(f"Pearsons correlation between {i} & {j} {pearsonr(results_df[i], results_df[j])}")

        
    
    
        df_added = df_last_gen[['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire', 'fitness']].copy()
        df_added["best"] = np.where(idx, 'BEST', 'OTHER')

        def plot_extra(x, y, **kwargs):
            sns.scatterplot(data=kwargs['data'], x=x.name, y=y.name, color=kwargs['color'])
            plt.legend(pearsonr(x, y))

        pg = sns.pairplot(df_added, hue='best', plot_kws={'alpha': 0.5},kind='scatter')

        pg.map_offdiag(plot_extra, color='orange', data=df_added[df_added['best']=='BEST'])
        new_title = ''
        pg._legend.set_title(new_title)
        # replace labels
        new_labels = ['all swarms','fittest swarm of generation 50']
        for t, l in zip(pg._legend.texts, new_labels):
            t.set_text(l)

        sns.move_legend(pg,"upper center",ncol=2)
        plt.subplots_adjust(top=0.97,right=0.98)
        plt.show()

def conv_plots():   
    environments = ["volcano", "lake", "random", "river",]
    results_directory = r"./results"
    fig, axs = plt.subplots(6, 4,sharex='col',sharey='row',figsize=(12,18))
    for e,env in enumerate(environments):

        df = pd.read_csv(os.path.join(results_directory, "output_"+env+".csv"))
        
        for v,val in enumerate(['Alignment', 'Cohesion', 'Separation', 'GoToWater', 'GoToFire', 'fitness']):
            if v==0:
                axs[v,e].set_title(env)
            sns.histplot(data=df, x="Gen", y=val,ax=axs[v,e],bins=50)
    plt.tight_layout()   
    plt.subplots_adjust(left=0.06,right=1,bottom=0.04,top=0.97,wspace=0,hspace=0)
    figurename = '2Dhist_feature_gen.pdf'
    plt.savefig(figurename, dpi=600)
    plt.show()

if __name__ == '__main__':
    main()