---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
execution:
  timeout: 240
---

# Exercise 3: electron vs gamma separation

## Motivation
```{figure} ./wouter_e_gamma.png
---
figclass: margin
---
Difference between an electron vs photon is at the start of the electromagnetic shower, where the photon has a gap. From Wouter Van De Pontseele, ICHEP 2020.
```
Electrons are visible in a LArTPC detector because of the electromagnetic showers that they trigger.

Photons, on the other hand, are neutral (no charge) and thus remain invisible to the LArTPC eyes until
they convert into electrons (pair production) or Compton scatter. In both cases, the visible outcome
will be an electromagnetic shower. 

How can we differentiate the two, then? The answer is in the very 
beginning of the EM shower. For an electron, this shower will be topologically connected to the interaction
vertex where the electron was produced. For a photon, there will be a *gap* (equal to the photon travel
path) until the EM shower start (when the photon becomes indirectly visible through pair production or
Compton scatter). That seems simple enough, right? Wrong, of course.

Energetic photons could interact at a distance short enough from the interaction vertex, that we would not
be able to see the gap. Or, the hadronic activity might be invisible, because it includes neutral particles
or because the particles are too low energy to be seen. In that case the interaction vertex might be hard to
identify, and the notion of a gap goes away too. For such cases, fortunately, there is another way to tell
electrons from gamma showers. Another major difference is in the energy loss rate at the start of the EM shower.
An electron would leave ionization corresponding to a single ionizing particle, whereas a pair of electron + positron
coming from a photon pair production would add up to two ionizing particle. Thus, we expect the $dE/dx$ at the
beginning of the shower to be roughly twice larger in the case of a gamma-induced shower compared to an electron-induced shower.

```{figure} ./wouter_dEdx.png
---
height: 300px
---
Example from MicroBooNE. Left is the shower $dE/dx$, right is the gap between the vertex and shower start. From Wouter Van De Pontseele, ICHEP 2020.
```

Why do we care? The difference becomes significant if, for example, you are looking for electron
neutrinos. One of the key signatures you would be looking for are electrons.

In this exercise, we will focus on finding the start of EM showers and computing the reconstructed $dQ/dx$ in these
segments. Optionally, you could compare that to the result of using automatic PID as predicted by the chain.

## Setup
If needed, you can edit the path to `lartpc_mlreco3d` library and to the data folder.
```{code-cell}
import os
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME') 
DATA_DIR = os.environ.get('DATA_DIR')
```

The usual imports and setting the right `PYTHON_PATH`...  click if you need to see them.
```{code-cell}
:tags: [hide-cell]

import sys, os
# set software directory
sys.path.insert(0, SOFTWARE_DIR)
```

```{code-cell}
:tags: [hide-cell]

import numpy as np
import yaml
import torch
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=False)

from mlreco.visualization import scatter_points, plotly_layout3d
from mlreco.visualization.gnn import scatter_clusters, network_topology, network_schematic
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.cluster.dense_cluster import fit_predict_np, gaussian_kernel
from mlreco.main_funcs import process_config, prepare
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.visualization.gnn import network_topology

from larcv import larcv
```

The configuration is loaded from the file [inference.cfg](../data/inference.cfg).
```{code-cell}
:tags: [hide-output]

cfg=yaml.load(open('%s/inference.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
cfg['iotool']['batch_size'] = 8 # customize batch size
# pre-process configuration (checks + certain non-specified default settings)
process_config(cfg)
# prepare function configures necessary "handlers"
hs=prepare(cfg)
```

The output is hidden because it reprints the entire (lengthy) configuration. Feel 
free to take a look if you are curious!

Finally we run the chain for 1 iteration:
```{code-cell}
# Call forward to run the net, store the output in "res"
data, output = hs.trainer.forward(hs.data_io_iter)
```
Now we can play with `data` and `output` to visualize what we are interested in. Feel free to change the
entry index if you want to look at a different entry!
```{code-cell}
entry = 4
```
Let us grab the interesting quantities:
```{code-cell}
clust_label = data['cluster_label'][entry]
input_data = data['input_data'][entry]
segment_label = data['segment_label'][entry][:, -1]

ghost_mask = output['ghost'][entry].argmax(axis=1) == 0
segment_pred = output['segmentation'][entry].argmax(axis=1)
```

## Step 1: Shower primary fragments
The shower clustering GNN also predicts for each fragment a binary classification
primary/non-primary fragment in `shower_node_pred`. We want to select primary shower fragments.
```{code-cell}
:tags: [hide-input]

shower_label = 0
primary_mask = output['shower_node_pred'][entry].argmax(axis=1) == 1
primary_shower_fragments = output['shower_fragments'][entry][primary_mask]

print('Found %d primary shower fragments in this entry' % len(primary_shower_fragments))
```
Each *fragment* is a list of voxel indices (indexing the original `data['input_data'][entry]`).
`primary_shower_fragments` is a list of fragments.

## Step 2: Identify the start
We will use PPN point predictions to figure out how to look at these primary shower
fragments and in which direction the particle was going. 
```{code-cell}
ppn = uresnet_ppn_type_point_selector(data['input_data'][entry], output, entry=entry,
                                      score_threshold=0.5, type_threshold=2)
ppn_voxels = ppn[:, :3]
ppn_score = ppn[:, 5]
ppn_type = ppn[:, 12]
ppn_endpoints = np.argmax(ppn[:, 13:], axis=1)

delta_label = 3
is_not_delta = ppn[:, 7 + delta_label] < 0.5
ppn_voxels = ppn_voxels[is_not_delta]
ppn_score = ppn_score[is_not_delta]
ppn_type = ppn_type[is_not_delta]
ppn_endpoints = ppn_endpoints[is_not_delta]

print("Found %d predicted PPN points" % len(ppn_voxels))
```
We can also take a quick look to see what the quality of PPN predictions are.

```{code-cell}
trace = []

trace+= scatter_points(input_data[ghost_mask],markersize=1,color=segment_pred[ghost_mask], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Predicted semantic labels (predicted no-ghost mask)'

trace += scatter_points(ppn_voxels, markersize=5, color=ppn_type, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3, hovertext=ppn_score)
trace[-1].name = "PPN predictions (w/ type prediction)"

trace += scatter_points(data['particles_label'][entry], markersize=5, color=data['particles_label'][entry][:, 4], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = "True point labels"

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

This will allow us to select
all the voxels of the primary fragment which are close to the shower start, i.e. within
some radius of the predicted PPN shower point. In case there was no PPN shower start 
point predicted, we also store the closest distance between the shower fragment and what
we think is a potential shower start. Later we can cut on that distance to eliminate
predicted shower starts that are actually far away, and were mistakenly matched.

```{code-cell}
:tags: [hide-input]

from scipy.spatial.distance import cdist

shower_point = ppn[is_not_delta][:, 7 + shower_label] > 0.5
print("Found %d potential shower start points" % np.count_nonzero(shower_point))
ppn_voxels = ppn_voxels[shower_point]

start_shower_fragments = []
start_distance = []
for frag in primary_shower_fragments:
    voxel_count = len(frag)
    d = cdist(ppn_voxels, data['input_data'][entry][ghost_mask][frag][:, :3])

    start_shower_fragments.append(d.argmin()//voxel_count)
    start_distance.append(d.min())
    
start_shower_fragments = np.array(start_shower_fragments)
start_distance = np.array(start_distance)
```



## Step 3: Use dQ/dx to separate
Now that we have a list of primary shower fragments and their start points,
it is time to select voxels near the start points and compute $dQ/dx$ in this
vicinity. 

```{code-cell}
distance_threshold = 5 # in voxels, to discard start points too far away
min_segment_size = 3 # in voxels
radius = 10 # in voxels
```
We also use a distance threshold to ignore shower fragments where
no PPN point was predicted nearby - we *could* try to use $dQ/dx$ to figure out
on which side the starting point is, but for the sake of this exercise, we will
just ignore them.
```{code-cell}
:tags: [hide-input]

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

segment_dQ, segment_dx, segment_dN = [], [], []

for frag_idx, frag in enumerate(primary_shower_fragments):
    if start_distance[frag_idx] > distance_threshold:
        continue
    d = cdist(data['input_data'][entry][ghost_mask][frag][:, :3], [ppn_voxels[start_shower_fragments[frag_idx]]])
    segment_mask = (d.reshape((-1,)) < radius)
    
    dQ = data['input_data'][entry][ghost_mask][frag][segment_mask][:, 4].sum()
    dN = np.count_nonzero(segment_mask)
    # Use PCA to compute dx
    projection = pca.fit_transform(data['input_data'][entry][ghost_mask][frag][segment_mask, :3])
    dx = projection[:, 0].max() - projection[:, 0].min()
    if dx < min_segment_size:
        continue
        
    segment_dQ.append(dQ)
    segment_dx.append(dx)
    segment_dN.append(dN)
    
segment_dQ = np.array(segment_dQ)
segment_dx = np.array(segment_dx)
segment_dN = np.array(segment_dN)

print("Kept %d / %d segments for dQ/dx" % (len(segment_dQ), len(primary_shower_fragments)))
```

## Step 4: let's see a plot
Now we can put it all together in a function `compute_dQdx(data, output, entry)`
that returns `segment_dQ, segment_dx, segment_dN`. Then run it on several entries,
several iterations to get a few entries for our final histogram!

```{code-cell}
:tags: [hide-cell]

def compute_dQdx(data, output, entry):
    clust_label = data['cluster_label'][entry]
    input_data = data['input_data'][entry]
    segment_label = data['segment_label'][entry][:, -1]

    ghost_mask = output['ghost'][entry].argmax(axis=1) == 0
    segment_pred = output['segmentation'][entry].argmax(axis=1)

    primary_mask = output['shower_node_pred'][entry].argmax(axis=1) == 1
    primary_shower_fragments = output['shower_fragments'][entry][primary_mask]
    
    ppn = uresnet_ppn_type_point_selector(data['input_data'][entry], output, entry=entry,
                                          score_threshold=0.5, type_threshold=2)
    ppn_voxels = ppn[:, :3]
    ppn_score = ppn[:, 5]
    ppn_type = ppn[:, 12]
    ppn_endpoints = np.argmax(ppn[:, 13:], axis=1)

    delta_label = 3
    is_not_delta = ppn[:, 7 + delta_label] < 0.5
    ppn_voxels = ppn_voxels[is_not_delta]
    ppn_score = ppn_score[is_not_delta]
    ppn_type = ppn_type[is_not_delta]
    ppn_endpoints = ppn_endpoints[is_not_delta]

    shower_point = ppn[is_not_delta][:, 7 + shower_label] > 0.5
    ppn_voxels = ppn_voxels[shower_point]

    start_shower_fragments = []
    start_distance = []
    for frag in primary_shower_fragments:
        voxel_count = len(frag)
        d = cdist(ppn_voxels, data['input_data'][entry][ghost_mask][frag][:, :3])

        start_shower_fragments.append(d.argmin()//voxel_count)
        start_distance.append(d.min())

    start_shower_fragments = np.array(start_shower_fragments)
    start_distance = np.array(start_distance)
    
    segment_dQ, segment_dx, segment_dN = [], [], []

    for frag_idx, frag in enumerate(primary_shower_fragments):
        if start_distance[frag_idx] > distance_threshold:
            continue
        d = cdist(data['input_data'][entry][ghost_mask][frag][:, :3], [ppn_voxels[start_shower_fragments[frag_idx]]])
        segment_mask = (d.reshape((-1,)) < radius)

        dQ = data['input_data'][entry][ghost_mask][frag][segment_mask][:, 4].sum()
        dN = np.count_nonzero(segment_mask)
        # Use PCA to compute dx
        projection = pca.fit_transform(data['input_data'][entry][ghost_mask][frag][segment_mask, :3])
        dx = projection[:, 0].max() - projection[:, 0].min()
        if dx < min_segment_size:
            continue

        segment_dQ.append(dQ)
        segment_dx.append(dx)
        segment_dN.append(dN)

    segment_dQ = np.array(segment_dQ)
    segment_dx = np.array(segment_dx)
    segment_dN = np.array(segment_dN)
    return segment_dQ, segment_dx, segment_dN
```

```{code-cell}
:tags: [hide-output]

iterations = 10

all_dQ, all_dx, all_dN = [], [], []
for iteration in range(iterations):
    data, output = hs.trainer.forward(hs.data_io_iter)
    for entry in range(len(data['input_data'])):
        print("Iteration %d / Entry %d " % (iteration, entry))
        segment_dQ, segment_dx, segment_dN = compute_dQdx(data, output, entry)
        all_dQ.extend(segment_dQ)
        all_dx.extend(segment_dx)
        all_dN.extend(segment_dN)
        
all_dQ = np.array(all_dQ)
all_dx = np.array(all_dx)
all_dN = np.array(all_dN)
```

```{code-cell}
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

plt.hist(all_dQ/all_dx, range=[0, 5000], bins=20)
plt.xlabel("dQ/dx")
plt.ylabel("Predicted primary shower fragments")
```
## Other readings