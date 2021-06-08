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

# Particle clustering

## Imports and configuration
The usual imports, click to see the details.

```{code-cell}
:tags: [hide-cell]

import numpy as np
import yaml
import torch
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=False)
```

```{code-cell}
:tags: [hide-cell]

import sys, os
# set software directory
software_dir = '%s/lartpc_mlreco3d' % os.environ.get('HOME')
sys.path.insert(0,software_dir)
```

```{code-cell}
:tags: [hide-cell]

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

And the configuration which is loaded from the file [inference.cfg](./inference.cfg):
```{code-cell}
:tags: [hide-output]

cfg=yaml.load(open('../data/inference.cfg', 'r'),Loader=yaml.Loader)
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
Now we can play with `data` and `output` to visualize what we are interested in.
```{code-cell}
entry = 0
```
Let us grab the interesting quantities:
```{code-cell}
clust_label = data['cluster_label'][entry]
input_data = data['input_data'][entry]
segment_label = data['segment_label'][entry][:, -1]

ghost_mask = output['ghost'][entry].argmax(axis=1) == 0
segment_pred = output['segmentation'][entry].argmax(axis=1)
```

## Track clustering

### Stage 1: SPICE
SPICE requires to set some tunable thresholds. Here we pick these to favor purity 
over efficiency, so expect to see many small fragments of tracks at this stage.

```{code-cell}
p_thresholds = {
    0: 0.95, #0.14,
    1: 0.95, #0.24,
    2: 0.95, #0.48,
    3: 0.95 #0.48
}
s_thresholds = {
    0: 0.0, #0.8,
    1: 0.0, #0.90,
    2: 0.0, #0.55,
    3: 0.35 #0.80
}
```
We also need to retrieve batch ids for the embeddings (this will be fixed in the future...).
```{code-cell}
batch_mask = []
count = 0
for e in range(len(data['input_data'])):
    batch_mask.append(np.ones((data['input_data'][e][output['ghost'][e].argmax(axis=1) == 0].shape[0],)) * e)
    count += data['input_data'][e].shape[0]
batch_mask = np.hstack(batch_mask)
```

Now, SPICE provides us with 3 outputs, embeddings, margins and seediness.
```{code-cell}
embeddings = np.array(output['embeddings'])[batch_mask == entry]
margins = np.array(output['margins'])[batch_mask == entry]
seediness = np.array(output['seediness'])[batch_mask == entry]
```
We only want to run SPICE on predicted track voxels:
```{code-cell}
c = 1
class_mask = segment_pred[ghost_mask] == c
```
Time to run the fit function that looks at the predicted embeddings and
does the actual clustering inference for us:
```{code-cell}
cluster_preds = fit_predict_np(embeddings = embeddings[class_mask],
                                          seediness = seediness[class_mask], 
                                          margins = margins[class_mask], 
                                          fitfunc = gaussian_kernel,
                                          s_threshold=s_thresholds[c],
                                          p_threshold=p_thresholds[c])
```

And visualization time!

```{code-cell}
trace = []

edep = data['input_data'][entry][ghost_mask]

trace+= scatter_points(clust_label[clust_label[:, -1] == c],
                       markersize=1,
                       color=clust_label[clust_label[:, -1] == c, -5],
                       hovertext=clust_label[clust_label[:, -1] == c, -5],
                       colorscale=plotly.colors.qualitative.Dark24)

trace[-1].name = 'True cluster labels'

trace+= scatter_points(edep[class_mask],
                       markersize=1,
                       color=cluster_preds,
                       hovertext=cluster_preds,
                       colorscale=plotly.colors.qualitative.Dark24)

trace[-1].name = 'Clustering predictions'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Stage 2: GNN
We use a GNN to then cluster these small track fragments together.
First we need to retrieve true labels that take into account the 
ghost points. Labels are assigned by nearest non-ghost voxel.

```{code-cell}
clust_label_adapted = adapt_labels(output, data['segment_label'], data['cluster_label'])[entry]

clust_ids_true = get_cluster_label(torch.tensor(clust_label_adapted), output['track_fragments'][entry], column=6)
clust_ids_pred = output['track_group_pred'][entry]
```

Then we can look at the visualization:
```{code-cell}
trace = []

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['track_fragments'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_true,
                         markersize=2, cmin=0, cmax=40, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Cluster truth'

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['track_fragments'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_pred,
                         markersize=2, cmin=0, cmax=40, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Cluster predictions'

fig = go.Figure(data=trace,layout=plotly_layout3d())
#fig = go.Figure(data=trace,layout=layout)
fig.update_layout(legend=dict(x=1.1, y=0.9))

iplot(fig)
```

Note that this uses a different helper function from `lartpc_mlreco3d`: the function
`network_topology` is useful to visualize the output of a GNN. Its first argument is 
the input data (to provide the voxel coordinates), its second argument is a list of 
list of voxel indices (list of fragments).

## Shower clustering

```{code-cell}
clust_label_adapted = adapt_labels(output, data['segment_label'], data['cluster_label'])[entry]

clust_ids_true = get_cluster_label(torch.tensor(clust_label_adapted), output['shower_fragments'][entry], column=6)
clust_ids_pred = output['shower_group_pred'][entry]
```

```{code-cell}
trace = []

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['shower_fragments'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_true,
                         markersize=2, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Cluster truth'

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['shower_fragments'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_pred,
                         markersize=2, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Cluster predictions'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.1, y=0.9))

iplot(fig)
```