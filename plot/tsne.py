from sklearn.manifold import TSNE
from subprocess import call
import matplotlib.colors as c
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import math
import os
import seaborn as sns
sns.set_theme(style="darkgrid")

class tSNE():
    def __init__(self, dim=2, x_train_predict=[], name='tsne', load=False, path=''):
        super(tSNE, self).__init__()
        self.dim = dim  
        self.name = name
        if not os.path.exists('./TSNE_images'):
            os.makedirs('./TSNE_images')
        if not load:
            print ("Performing t-SNE dimensionality reduction...")
            features = TSNE(n_components=self.dim).fit_transform(x_train_predict)
            x_min, x_max = np.min(features, 0), np.max(features, 0)
            self.x_train_encoded = (features - x_min) / (x_max - x_min)
            np.save('./TSNE_images/{}.npy'.format(self.name), self.x_train_encoded)
            print ("Saved and Done.")
        else:
            self.x_train_encoded = np.load(path)

    def tsne_plt(self, y_train):

        x_train_encoded = self.x_train_encoded
        cmap = plt.get_cmap('ocean', 10)

        # 3-dim vis: show one view, then compile animated .gif of many angled views
        if self.dim == 3:
            # Simple static figure
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                    c=y_train, cmap=cmap, edgecolor='black')
            fig.colorbar(p, drawedges=True)
            
            plt.savefig('./TSNE_images/{}_3d.png'.format(self.name), bbox_inches='tight', dpi=200)
            plt.show()

        # 2-dim vis: plot and colorbar.
        elif self.dim == 2:
            fig = plt.figure()
            ax = plt.axes()
            scatter = ax.scatter(x_train_encoded[:,0], x_train_encoded[:,1], alpha=1,
                                  c=y_train, s=20, edgecolor='black', cmap=cmap)
            
            legend1 = ax.legend(*scatter.legend_elements(), loc="upper right",
                                bbox_to_anchor=(1.2, 1), title="Classes", borderaxespad=0.0)
            #for i, cl in enumerate(classes):
            #    legend1.get_texts()[i].set_text(cl)
            
            ax.add_artist(legend1)
            plt.axis('off')
            #plt.colorbar(drawedges=True)
            plt.savefig('./TSNE_images/{}_tsne.png'.format(self.name),
                        bbox_inches='tight', dpi=300)
            plt.savefig('./TSNE_images/{}_tsne_eps.eps'.format(self.name),
                        bbox_inches='tight')