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
  timeout: 60
---

# Interaction clustering, PID and primary particles

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

## Visualization of interaction clustering

```{code-cell}
clust_label_adapted = adapt_labels(output, data['segment_label'], data['cluster_label'])[entry]

clust_ids_true = get_cluster_label(torch.tensor(clust_label_adapted), output['particles'][entry], column=7)
clust_ids_pred = output['inter_group_pred'][entry]
```

```{code-cell}
trace = []

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['particles'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_true,
                         markersize=2, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'True interactions'


trace+= scatter_points(clust_label_adapted,markersize=1,color=clust_label_adapted[:, 7], colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Adapted cluster labels'

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['particles'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=clust_ids_pred,
                         markersize=2, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Predicted interactions'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.1, y=0.9))

iplot(fig)
```

## Primary particles predictions
We need to get the true labels first:
```{code-cell}
kinematics_label = data['kinematics_label'][entry]
true_vtx, inv = np.unique(kinematics_label[:, 9:12], axis=0, return_index=True)
true_vtx_primary = kinematics_label[inv, 12]
```
And the predictions:
```{code-cell}
vtx_primary_pred = output['node_pred_vtx'][entry][:, 3:].argmax(axis=1)
```

```{code-cell}
trace = []

trace+= scatter_points(kinematics_label,markersize=1,color=kinematics_label[:, 12], colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'True vertex primary particles'

trace += network_topology(data['input_data'][entry][ghost_mask],
                         output['particles'][entry],
                         #edge_index=output['frag_edge_index'][entry],
                         clust_labels=vtx_primary_pred,
                         markersize=2, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Predicted vertex primary particles'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.1, y=0.9))

iplot(fig)
```

## Particle identification (PID)
The predictions are in `node_pred_type`:
```{code-cell}
type_pred = output['node_pred_type'][entry].argmax(axis=1)
```

```{code-cell}
trace = []

trace+= scatter_points(clust_label,markersize=1,color=kinematics_label[:, -2], colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'True particle type'

trace+= scatter_points(clust_label,markersize=1,color=type_pred, colorscale=plotly.colors.qualitative.Dark24)
trace[-1].name = 'Predicted particle type'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.1, y=0.9))

iplot(fig)
```