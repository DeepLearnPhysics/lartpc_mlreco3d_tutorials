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
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor

from larcv import larcv
```

The configuration is loaded from the file [inference.cfg](../data/inference.cfg).
```{code-cell}
:tags: [hide-output]

cfg=yaml.load(open('%s/inference.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
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

### Stage 1: GraphSPICE
GraphSPICE is a first pass of CNN-based voxel clustering. We optimize this round for purity, hence there might be many small track fragments predicted by this pass.
```{code-cell}
graph = output['graph'][0]
graph_info = output['graph_info'][0]
gs_manager = ClusterGraphConstructor(cfg['model']['modules']['graph_spice']['constructor_cfg'], graph_batch=graph, graph_info=graph_info)

track_label = 1
```

Time to run the fit function that looks at the predicted embeddings and
does the actual clustering inference for us:
```{code-cell}
pred, G, subgraph = gs_manager.fit_predict_one(entry, gen_numpy_graph=True)
```

And visualization time!

```{code-cell}
trace = []

edep = input_data[segment_label < 5]

trace+= scatter_points(clust_label[clust_label[:, -1] == track_label],markersize=1,color=clust_label[clust_label[:, -1] == track_label, 6], cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True cluster labels (true no-ghost mask)'

clust_label_adapted = adapt_labels(output, data['segment_label'], data['cluster_label'])[entry]

trace+= scatter_points(clust_label_adapted[segment_pred[ghost_mask] == track_label],markersize=1,color=clust_label_adapted[segment_pred[ghost_mask] == track_label, 6], cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True cluster labels (predicted no-ghost mask & semantic)'

trace+= scatter_points(subgraph.pos.cpu().numpy(),markersize=1,color=pred, cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Predicted clusters (predicted no-ghost mask)'

# trace+= scatter_points(clust_label,markersize=1,color=output['graph_spice_label'][entry][:, 5], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
# trace[-1].name = 'Input to graph spice'

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