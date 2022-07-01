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
  timeout: 180
---

# Semantics & points of interest

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
Now we can play with `data` and `output` to visualize what we are interested in.

## Semantic segmentation (UResNet)
Let us take a look at the first entry. Feel free to change the
entry index if you want to look at a different entry!
```{code-cell}
entry = 0
```
We extract quantities of interest from the `data` and `output` dictionaries.
Here, we want the `input_data` (voxel coordinates and corresponding reconstructed energy depositions)
and `segment_label` (true semantic labels for each voxel). We will use the predicted `ghost_mask`
(binary mask ghost / non-ghost voxel) and the UResNet predictions `segment_pred` are obtained
from `output['segmentation']` (softmax scores).
```{code-cell}
input_data = data['input_data'][entry]
segment_label = data['segment_label'][entry][:, -1]

ghost_mask = output['ghost'][entry].argmax(axis=1) == 0
segment_pred = output['segmentation'][entry].argmax(axis=1)
```

We use Plotly to visualize the result:
```{code-cell}
trace = []

edep = input_data[segment_label < 5]

trace+= scatter_points(input_data[segment_label < 5],markersize=1,color=segment_label[segment_label < 5], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True semantic labels (true no-ghost mask)'

trace+= scatter_points(input_data[ghost_mask],markersize=1,color=segment_pred[ghost_mask], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Predicted semantic labels (predicted no-ghost mask)'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

## Points of interest (PPN)
PPN makes a prediction for each non-zero voxel. We need to apply a post-processing function
to apply the predicted attention mask and bring down the number of point proposals.

```{code-cell}
ppn = uresnet_ppn_type_point_selector(data['input_data'][entry], output, entry=entry,
                                      score_threshold=0.5, type_threshold=2)
ppn_voxels = ppn[:, 1:4]
ppn_score = ppn[:, 5]
ppn_type = ppn[:, 12]
ppn_endpoints = np.argmax(ppn[:, 13:], axis=1)
```
The columns of `ppn` contain in this order:

- point coordinates x, y, z
- batch id
- detection score (2 softmax values)
- occupancy (how many points were merge to this single point during post processing)
- softmax scores for 5 semantic types
- type prediction (max softmax score among 5 semantic types)

You can also use the softmax scores for the 5 semantic types to make finer point type predictions - for example at a vertex, you can expect these scores to be high for two or more types.

We remove points that have a high score for being Delta rays starting points:
```{code-cell}
delta_label = 3
is_not_delta = ppn[:, 7 + delta_label] < 0.5
ppn_voxels = ppn_voxels[is_not_delta]
ppn_score = ppn_score[is_not_delta]
ppn_type = ppn_type[is_not_delta]
ppn_endpoints = ppn_endpoints[is_not_delta]
```

And at last! We can visualize both the true and predicted points:
```{code-cell}
trace = []

trace+= scatter_points(input_data[ghost_mask],markersize=1,color=segment_pred[ghost_mask], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Predicted semantic labels (predicted no-ghost mask)'

trace += scatter_points(ppn_voxels, markersize=5, color=ppn_type, cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3, hovertext=ppn_score)
trace[-1].name = "PPN predictions (w/ type prediction)"

trace += scatter_points(ppn_voxels[ppn_type == 1], markersize=5, color=ppn_endpoints[ppn_type == 1], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3, hovertext=ppn_endpoints)
trace[-1].name = "PPN predictions (start/end)"

trace += scatter_points(data['particles_label'][entry], markersize=5, color=data['particles_label'][entry][:, 4], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = "True point labels"

trace += scatter_points(data['particles_label'][entry], markersize=5, color=data['particles_label'][entry][:, 6], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = "True point labels (start/end)"

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```
The color of the points corresponds to either their semantic type, or a binary start/end classification.

---

This is all there is to know about UResNet + PPN output and its visualization.
