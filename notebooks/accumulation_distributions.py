#+title: Accumulation Distributions
#+PROPERTY: header-args:python :python /home/jo/micromamba/envs/torchland/bin/python :session flowaccstats
#If a landscape is fractal, would we expect the distribution of flow accumulation to be roughly a powerlaw?  How similar would the flow accumulation distributions be?

#+begin_src python
import numpy as np
import sqlite3
import os
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import matplotlib.pyplot as plt

numpy_dir = "../model_run_flowacc"
db = "../model_runs.db"
#+end_src

#+begin_src python
connection = sqlite3.connect(db)
seed_df = pd.read_sql("SELECT model_run_id, \"model_param.seed\" FROM model_run_params", connection)
seed = seed_df['model_param.seed'][0]
one_seed_df = seed_df[seed_df['model_param.seed'] == seed]
names = one_seed_df['model_run_id'].tolist()
#+end_src


#+begin_src python :results output
if not os.path.exists("products/flow_accumulation_distances.npy"):
    all_data = {}
    for file in os.listdir(numpy_dir):
        if not os.path.splitext(file)[0] in names:
            continue
        data = np.load(os.path.join(numpy_dir, file))
        flow_accumulation = data[5:-5, 5:-5]
        flow_accumulation = flow_accumulation.flatten()
        all_data[os.path.splitext(file)[0]] = flow_accumulation
    print("loaded Data")
    ks_metric = lambda u, v: ks_2samp(u, v).statistic
    distance_vector = pdist([v for v in all_data.values()], metric=ks_metric)
    np.save("products/flow_accumulation_distances.npy", distance_vector)
else:
    distance_vector = np.load("products/flow_accumulation_distances.npy")
    print("loaded distances")
#+end_src

#+begin_src python
if os.path.exists("products/flow_accumulation_clusters5.npy"):
    clusters = np.load("products/flow_accumulation_clusters5.npy")
else:
    print("Calculating clusters")
    linkage_matrix = linkage(distance_vector, method='ward')
    clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')
    np.save("products/flow_accumulation_clusters5.npy", clusters)
#+end_src

#+begin_src python :results file :exports both
connection = sqlite3.connect(db)
k_d_df = pd.read_sql("SELECT model_run_id, \"model_param.streampower.k\", \"model_param.diffuser.D\", \"model_param.seed\" FROM model_run_params",connection)
k_d_df = k_d_df[k_d_df['model_param.seed']==seed]
k_d_df['cluster'] = clusters
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(k_d_df['model_param.streampower.k'], k_d_df['model_param.diffuser.D'], c=k_d_df['cluster'], cmap='viridis', s=50, alpha=0.7)
fig.legend(handles=scatter.legend_elements()[0], labels=sorted(k_d_df['cluster'].unique()), title="Clusters")
ax.set_xlabel('Streampower K')
ax.set_ylabel('Diffusivity')
ax.set_title('Flow Accumulation Clustered by K-S Distance')
fig.savefig("figs/flow_accumulation_clusters.png")
'figs/flow_accumulation_clusters5.png'
#+end_src
