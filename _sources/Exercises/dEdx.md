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

# Exercise 2: muon $dE/dx$

## MIP and Bragg peak
Minimally ionizing particles (MIP) are charged particles which lose energy when passing through matter at a rate close to minimal.
Particles such as muons often have energy losses close to the MIP level and are treated in practice as MIP. The only exception is
when the muon comes to a stop and experiences a Bragg peak.

```{figure} ./bragg_peak.png
---
height: 200px
---
Example of muon Bragg peak. The muon is travelling from bottom left to top right. The color scale represents energy deposition. Red means more energy deposited. The sudden increase in deposited (lost) energy towards the end of the muon trajectory is the Bragg peak. From MicroBooNE (arxiv: 1704.02927)
```

## Motivation
We know that the energy loss rate of a MIP in argon is ~2 MeV/cm. Hence our goal is to carefully isolate the MIP-like sections of
muons (i.e. leaving out the ends of the track), and compute the (reconstructed) energy loss along these trajectories $dQ/dx$.
This can inform the detector calibration effort, for example, since we can compare the peak of the $dQ/dx$ histogram with the theoretical
expected value of ~2 MeV/cm (although there are many detector effects that make this not straightforward). We can also study the spatial uniformity
of the detector by looking at MIP $dQ/dx$ values in different regions of the detector, etc.

## Setup
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

cfg=yaml.load(open('./inference.cfg', 'r'),Loader=yaml.Loader)
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


## Step 1: find muons
We could first look at the semantic segmentation (UResNet) outputs, specifically at the predicted track voxels.
Then, we could just run a simple clustering algorithm such as DBSCAN on these predictions (as in Exercise 1 with Michel electrons). 
BUT! Here we will be ambitious 🚀 and attempt to use higher level predictions from the reconstruction chain instead.

### Track-like particles
 The chain outputs a list of particles and their corresponding predicted 
semantic type, so let's retrieve the track-like particles.

```{code-cell}
track_label = 1
output['particles'][entry][output['particles_seg'][entry] == track_label]
```

### PID predictions
Then, the particle type identification stage lets us select among these track-like particles the ones that are predicted to be muons.

## Step 2: muon direction and binning
This step is more of a Scipy/Numpy exercise! Use something like PCA to compute the general direction of the muon track, and bin the voxels
along this track to define segments of a certain length:

```{code-cell}
segment_length = 7 # in voxels, that's about 2.1cm in our dataset (3mm/voxel)
```



## Step 3: plot dQ/dx histogram
For each segment identified, sum the reconstructed energy deposited ($dQ$) and divide by the segment length ($dx$) to obtain $dQ/dx$.
Make a histogram with all the segments identified. Hopefully, we can see a nice peak shape...

## More readings

* Passage of particles through matter from the Particle Data Group, https://pdg.lbl.gov/2005/reviews/passagerpp.pdf.