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

# Exercise 2: muon stopping power

The *stopping power* of a particle usually refers to the energy loss rate $dE/dx$ when it passes through matter. When charged particles
travel through our LArTPC detector, they interact with the argon and lose energy.

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
expected value (although there are many detector effects that make this not straightforward). We can also study the spatial uniformity
of the detector by looking at MIP $dQ/dx$ values in different regions of the detector, etc.

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


## Step 1: find muons
We could first look at the semantic segmentation (UResNet) outputs, specifically at the predicted track voxels.
Then, we could just run a simple clustering algorithm such as DBSCAN on these predictions (as in Exercise 1 with Michel electrons). 

BUT! Here we will be ambitious 🚀 and attempt to use higher level predictions from the reconstruction chain instead.

### Track-like particles
 The chain outputs a list of particles in `output['particles']` and their corresponding predicted 
semantic type in `output['particles_seg']`, so let's retrieve the track-like particles.

```{code-cell}
track_label = 1
track_mask = output['particles_seg'][entry] == track_label
print("We have %d track particles detected." % np.count_nonzero(track_mask))
```

### PID predictions
Then, the particle type identification stage lets us select among these track-like particles the ones that are predicted to be muons.
You can access the PID predictions via `output['node_pred_type']` (shape $N_{particles} x N_{types}$, contains softmax scores).

```{code-cell}
muon_mask = np.argmax(output['node_pred_type'][entry], axis=1) == 2
print("We have %d particles predicted to be muons." % np.count_nonzero(muon_mask))

muon_particles = output['particles'][entry][track_mask & muon_mask]
print("We selected %d particles as candidate muons." % len(muon_particles))
```

## Step 2: muon direction and binning
This step is more of a Scipy/Numpy exercise! Use PCA to compute the general direction of the muon track, and bin the voxels
(`np.digitize`) along this track to define segments of a certain length.

For each segment identified, sum the reconstructed energy deposited ($dQ$) and compute the segment length ($dx$).
You can add some cuts such as minimal number of voxels in a segment to remove problematic segments (e.g. if there are any
gaps in the track, etc).

```{code-cell}
segment_length = 7 # in voxels, that's about 2.1cm in our dataset (3mm/voxel)
min_voxels = 5
edge_threshold = 12 # in voxels
```

Write a function `compute_dQdx(entry)` that does all of this and returns three arrays of values `segment_dQ, segment_dx, segment_dN`.
You can clik on the button on the right "Click to show" to see the basic solution that we wrote (by no means the only possible one!).

```{code-cell}
:tags: [hide-cell]

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

def compute_dQdx(entry):
    track_mask = output['particles_seg'][entry] == track_label
    muon_mask = np.argmax(output['node_pred_type'][entry], axis=1) == 2
    muon_particles = output['particles'][entry][track_mask & muon_mask]
    
    segment_dQ = []
    segment_dx = []
    segment_dN = []

    for part in muon_particles:
        projected_coords = pca.fit_transform(data['input_data'][entry][part][:, :3])
        #print(projected_coords[:, 0])

        # Exclude edges
        main_body = (projected_coords[:, 0] > projected_coords[:, 0].min() + edge_threshold) & (projected_coords[:, 0] < projected_coords[:, 0].max() - edge_threshold)
        projected_coords = projected_coords[main_body]

        # Bin the track into segments
        bins = np.arange(np.floor(projected_coords[:, 0].min()), np.ceil(projected_coords[:, 0].max()), segment_length)
        idx = np.digitize(projected_coords[:, 0], bins)
        for i in np.unique(idx):
            seg_idx = idx == i
            dQ = data['input_data'][entry][part][main_body][seg_idx, 4].sum()
            dx = projected_coords[seg_idx, 0].max() - projected_coords[seg_idx, 0].min()
            dN = np.count_nonzero(seg_idx)
            if dx > 0 and dN > min_voxels:
                segment_dQ.append(dQ)
                segment_dx.append(dx)
                segment_dN.append(dN)

    segment_dQ = np.array(segment_dQ)
    segment_dx = np.array(segment_dx)
    segment_dN = np.array(segment_dN)
    return segment_dQ, segment_dx, segment_dN
```

Nice job! The hardest is done, now is time to reap the fruits of your work. We run this function `compute_dQdx` on all the entries
in this batch.

```{code-cell}
print("Running on %d entries... " % len(data['input_data']))

dQ, dx, dN = [], [], []
for entry in range(len(data['input_data'])):
    segment_dQ, segment_dx, segment_dN = compute_dQdx(entry)
    dQ.append(segment_dQ)
    dx.append(segment_dx)
    dN.append(segment_dN)
dQ = np.hstack(dQ)
dx = np.hstack(dx)
dN = np.hstack(dN)

print("... Done.")
```

## Step 3: plot dQ/dx histogram
Make a histogram with all the segments identified. Hopefully, we can see a nice peak shape...

```{code-cell}
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

plt.hist(dQ / dx, histtype='step', bins=20, range=[0, 5000])
plt.xlabel('dQ/dx (reconstructed charge / voxels distance)')
plt.ylabel('(Predicted) Muon Track Segments')
```

## Optional: display the PCA !
How well does the PCA reflect the muon track direction? Is our segment length reasonable w.r.t the track thickness?
If you want to study this in more details, looking at some visualizations can be helpful to get a first intuition.

## More readings

* Passage of particles through matter from the Particle Data Group, https://pdg.lbl.gov/2005/reviews/passagerpp.pdf.