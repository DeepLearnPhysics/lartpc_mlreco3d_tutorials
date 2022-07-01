---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
execution:
  timeout: 300
---

# Exercise 2: Muon stopping power¶

The *stopping power* of a particle usually refers to the energy loss rate $dE/dx$ when it passes through matter. When charged particles travel through our LArTPC detector, they interact with the argon and lose energy.

```{note}
Currently this tutorial only computes dQ/dx distribution. Need refresh to make a 2D plot of dQ/dx vs residual range.
```

## MIP and Bragg peak¶

Minimally ionizing particles (MIP) are charged particles which lose energy when passing through matter at a rate close to minimal. Particles such as muons often have energy losses close to the MIP level and are treated in practice as MIP. The only exception is when the muon comes to a stop and experiences a Bragg peak.

```{figure} ./bragg_peak.png
---
height: 200px
---
Example of muon Bragg peak. The muon is travelling from bottom left to top right. The color scale represents energy deposition. Red means more energy deposited. The sudden increase in deposited (lost) energy towards the end of the muon trajectory is the Bragg peak. From MicroBooNE (arxiv: 1704.02927)
```

## I. Motivation

We know that the energy loss rate of a MIP in argon is ~2 MeV/cm. Hence our goal is to carefully isolate the MIP-like sections of muons (i.e. leaving out the ends of the track), and compute the (reconstructed) energy loss along these trajectories $dQ/dx$. This can inform the detector calibration effort, for example, since we can compare the peak of the $dQ/dx$ histogram with the theoretical expected value (although there are many detector effects that make this not straightforward). We can also study the spatial uniformity of the detector by looking at MIP $dQ/dx$ values in different regions of the detector, etc. If we plot the dQ/dx change along the so-called "residual range" (i.e. distance to the end of the muon trajectory), we get a characteristic plot (due to the Bragg peak). In this tutorial we will focus on reproducing this plot.

```{figure} ./residual_range.png
---
height: 200px
---
Example of what we expect for a muon dQ/dx versus residual range 2d histogram. The sudden increase in deposited (lost) energy towards the end of the muon trajectory (= low residual range) is the Bragg peak. From MicroBooNE (arxiv: 2010.02390)
```

## II. Setup

Again, we start by setting our working environment. Some necessary boilerplate code:

+++

### a. Software and data directory

```{code-cell} ipython3
import os, sys
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME')
DATA_DIR = os.environ.get('DATA_DIR')
# Set software directory
sys.path.append(SOFTWARE_DIR)
```

+++ {"tags": []}

### b. Numpy, Matplotlib, and Plotly for Visualization and data handling.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import yaml

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=False)
```

### c. MLRECO specific imports for model loading and configuration setup
The imports below load some auxiliary functions and classes required to run the full chain in interactive mode.

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
from mlreco.main_funcs import process_config, prepare
```
Let’s load the configuration file `inference.cfg` to setup the full chain architecture and weights. This file uses the keyword `DATA_DIR` to symbolize the path to validation set and checkpoint file. We need to replace it with the actual location defined previously.
```{code-cell} ipython3
cfg = yaml.load(open('%s/inference.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
```
To keep things simple we will ask the chain to stay quiet and we can load a bigger dataset than the 100-events default one:
```{code-cell} ipython3
cfg['model']['modules']['chain']['verbose'] = False
# cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/mpvmpr_012022/test.root']
```
So far `cfg` was simply a dictionary loaded from a YAML configuration file. It needs to be consumed by a helper function `process_config` to be ready for the full chain usage:
```{code-cell} ipython3
process_config(cfg, verbose=False)
```

### d. Initialize and load weights to model using Trainer.
This next cell loads the dataset and the model to the notebook environment. `hs` stands for "handlers". It contains various useful access points, such as the I/O iterator `hs.data_io_iter` to directly access the dataset or the trainer instance `hs.trainer` which will enable us to actually *run* the network.

```{code-cell} ipython3
# prepare function configures necessary "handlers"
hs = prepare(cfg)
# Optionally, you can specifiy a list of images by listing the image entry ID numbers:
# hs = prepare(cfg, event_list=[0, 1, 2])
dataset = hs.data_io_iter
```

As usual, the model is now ready to be used (check for successful weight loading). Let's do one forward iteration to retrieve a handful of events.

```{code-cell} ipython3
data, result = hs.trainer.forward(dataset)
```

## III. Setup the analysis tools

The particles in a given image are classified into one of five particle types: photon $\gamma$, electron $e$, muon $\mu$, pion $\pi$, and proton $p$. Obtaining particles from the full chain is quite simple: we initialize the `FullChainEvaluator` for this batch of events and examine the particle composition through the `get_particles` (for true particles, `get_true_particles`) method.

```{code-cell} ipython3
from analysis.classes.ui import FullChainEvaluator
```

```{code-cell} ipython3
# Only run this cell once!
predictor = FullChainEvaluator(data, result, cfg, deghosting=True)
```

```{code-cell} ipython3
entry = 7    # Batch ID for current sample
```

## IV. Visualize true and predicted particles

Let’s first import plotting functions from `lartpc_mlreco3d` for easier visualization:

```{code-cell} ipython3
from mlreco.visualization import scatter_points, scatter_cubes, plotly_layout3d
from mlreco.visualization.plotly_layouts import white_layout, trace_particles, trace_interactions, dualplot
```
The evaluator will handle ghost masking internally, so all you have to do is retrieve the predicted and true semantic segmentation labels for visualization and analysis:
```{code-cell} ipython3
pred = predictor.get_predicted_label(entry, 'segment')
truth = predictor.get_true_label(entry, 'segment')
points = predictor.data_blob['input_data'][entry]
# Check if dimensions agree
assert (pred.shape[0] == truth.shape[0])
```
Let’s plot the voxel-level true and predicted semantic labels side-by-side:
```{code-cell} ipython3
trace1, trace2 = [], []
edep = points[:, 5]

trace1 += scatter_points(points,
                        markersize=1,
                        color=truth,
                        cmin=0, cmax=5, colorscale='rainbow')

trace2 += scatter_points(points,
                        markersize=1,
                        color=pred,
                        cmin=0, cmax=5, colorscale='rainbow')

fig = dualplot(trace1, trace2, titles=['True semantic labels (true no-ghost mask)', 'Predicted semantic labels (predicted no-ghost mask)' ])

iplot(fig)
```
By default, the label for tracks and michel electrons are 1 and 2, respectively.

```{code-cell} ipython3
track_label = 1
michel_label = 2
```

```{code-cell} ipython3
particles = predictor.get_particles(entry, only_primaries=False)
true_particles = predictor.get_true_particles(entry, only_primaries=False, verbose=False)
```

```{code-cell} ipython3
trace1 = trace_particles(particles, color='semantic_type')
trace2 = trace_particles(true_particles, color='semantic_type')
```

```{code-cell} ipython3
fig = dualplot(trace1, trace2, titles=['Predicted particles (predicted no-ghost mask)', 'True particles (predicted no-ghost mask)'])

iplot(fig)
```


The predicted particles, each color-coded according to its semantic type, will be displayed in left; the true particles on right.

## Step 1: Select stopping muons
We will select track-like predicted particles that are close to a Michel predicted particle.
For the stopping muon residual range study purpose, purity is more important than efficiency.
Hence we only select stopping muons that decay into a Michel electron.

```{code-cell} ipython3
from scipy.spatial.distance import cdist

def get_stopping_muons(particles, Michel_threshold=10):
    selected_muons = []
    closest_points = []

    # Loop over predicted particles
    for p in particles:
        if p.semantic_type != track_label: continue
        coords = p.points

        # Check for presence of Michel electron
        attached_to_Michel = False
        closest_point = None
        for p2 in particles:
            if p2.semantic_type != 2: continue
            d =  cdist(p.points, p2.points)
            if d.min() >= Michel_threshold: continue
            attached_to_Michel = True
            closest_point = d.min(axis=1).argmin()

        if not attached_to_Michel: continue

        selected_muons.append(p)
        closest_points.append(closest_point)

    return selected_muons, closest_points
```

```{code-cell} ipython3
selected_muons, closest_points = get_stopping_muons(particles)
selected_muons
```
And just as a sanity check, let's run the same function on the true particles to ensure it selects what we expect:
```{code-cell} ipython3
get_stopping_muons(true_particles)
```

## Step 2: Muon direction with PCA
Once we have a suitable muon selected, the next step is to bin it. To make the binning easier, we will find the approximate direction of the muon using principal component analysis (PCA) then project all muon voxels along this axis.

```{code-cell} ipython3
from sklearn.decomposition import PCA

def get_PCA(particles):
    pca = PCA(n_components=2)
    pca_coordinates = []
    pca_directions = []
    for p in particles:
        coords = p.points
        # PCA to get a rough direction
        coords_pca = pca.fit_transform(p.points)[:, 0]
        pca_coordinates.append(coords_pca)
        pca_directions.append(pca.components_[0, :])
    return pca_coordinates, pca_directions
```
Time to run our function on the muon we just selected:
```{code-cell} ipython3
pca_coordinates, pca_directions = get_PCA(selected_muons)
```
It is always a good idea to visualize the output of each intermediate stage to make sure the output is not garbage for the next step. The muon voxels are in blue, the computed PCA direction is in orange.
```{code-cell} ipython3
trace = []

for idx, (coords, direction) in enumerate(zip(pca_coordinates, pca_directions)):
    trace += scatter_points(selected_muons[idx].points,
                            markersize=1,
                            color='blue')
    # Artificially create a point cloud for the PCA direction visualization
    pca_points = selected_muons[idx].points[coords.argmin()][None, :] + direction[None, :] * (coords[:, None]-coords.min())
    trace += scatter_points(pca_points,
                           markersize=1,
                           color='orange')


fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.layout.scene.xaxis.range=[0, 768]
fig.layout.scene.yaxis.range=[0, 768]
fig.layout.scene.zaxis.range=[0, 768]
iplot(fig)
```
The orange direction seems to match *roughly* the muon direction, so we can move on to the next step.

## Step 3: Local dQ/dx

Using the rough muon direction, we will define segments by binning the projection of all voxels onto this axis.

```{code-cell} ipython3
def get_dQdx(particles, closest_points, pca_coordinates,
            bin_size=17,
            return_pca=False): # about5cm

    df = {
        "dQ": [],
        "dx": [],
        "residual_range": []
    }

    pca = PCA(n_components=2)

    list_coords, list_pca, list_directions = [], [], []
    for p, closest_point, coords_pca in zip(particles, closest_points, pca_coordinates):
        coords = p.points

        # Make sure where the end vs start is
        # if end == 0 we have the right bin ordering, otherwise might need to flip when we record the residual range
        distances_endpoints = [((coords[coords_pca.argmin(), :] - coords[closest_point, :])**2).sum(), ((coords[coords_pca.argmax(), :] - coords[closest_point, :])**2).sum()]
        end = np.argmin(distances_endpoints)

        # Split into segments and compute local dQ/dx
        bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
        bin_inds = np.digitize(coords_pca, bins)

        for i in np.unique(bin_inds):
            mask = bin_inds == i
            if np.count_nonzero(mask) < 2: continue
            # Repeat PCA locally for better measurement of dx
            pca_axis = pca.fit_transform(p.points[mask])
            list_coords.append(p.points[mask])
            list_pca.append(pca_axis[:, 0])
            list_directions.append(pca.components_[0, :])

            dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
            dQ = p.depositions[mask].sum()
            residual_range = (i if end == 0 else len(bins)-i-1) * bin_size

            df["dx"].append(dx)
            df["dQ"].append(dQ)
            df["residual_range"].append(residual_range)

    df = pd.DataFrame(df)

    if return_pca:
        return list_coords, list_pca, list_directions

    return df     
```

For each segment, we compute
* dQ (sum of reconstructed charge of all voxels in the segment)
* dx (for better precision we recompute a local PCA direction on the segment voxels exclusively)
* residual range (distance from the muon end, well-defined using the Michel contact point)

```{code-cell} ipython3
get_dQdx(selected_muons, closest_points, pca_coordinates)
```

Once again, let's take the time to visualize the binning and make sure the segments make sense. The black voxels belong to the local PCA directions, the colored voxels belong to different segments (color = segment id).

```{code-cell} ipython3
import matplotlib.pyplot as plt
trace = []

list_coords, list_pca, list_directions = get_dQdx(selected_muons, closest_points, pca_coordinates, return_pca=True)
cmap = plt.cm.get_cmap('Set1')
n = len(list_coords)
for idx, (coords, pca_axis, direction) in enumerate(zip(list_coords, list_pca, list_directions)):
    trace += scatter_points(coords,
                            markersize=2,
                            color='rgb%s' % str(cmap(idx/n)[:-1]))
    #print(direction[None, :] * coords[:, None])
    #print(cmap(idx/n))
    pca_points = coords[pca_axis.argmin()][None, :] + direction[None, :] * (pca_axis[:, None]-pca_axis.min())
    trace += scatter_points(pca_points,
                           markersize=2,
                           color='black')


fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.layout.scene.xaxis.range=[0, 768]
fig.layout.scene.yaxis.range=[0, 768]
fig.layout.scene.zaxis.range=[0, 768]
iplot(fig)
```

## Step 4: Repeat with high statistics
We have computed everything we want, we just need to repeat this with higher statistics.

```{code-cell} ipython3
muons = pd.DataFrame({
        "index": [],
        "dQ": [],
        "dx": [],
        "residual_range": []
    })

from tqdm import tqdm
for iteration in tqdm(range(10)):
    data, result = hs.trainer.forward(dataset)
    predictor = FullChainEvaluator(data, result, cfg, deghosting=True, processor_cfg={'michel_primary_ionization_only': True})
    for entry, index in enumerate(predictor.index):  
        particles = predictor.get_particles(entry, only_primaries=False)
        selected_muons, closest_points = get_stopping_muons(particles)
        pca_coordinates, pca_directions = get_PCA(selected_muons)
        df = get_dQdx(selected_muons, closest_points, pca_coordinates)
        df['index'] = index
        muons = pd.concat([muons, df])
```
```{code-cell} ipython3
muons
```

## Plot dQ/dx vs residual range
```{code-cell} ipython3
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')
```

```{code-cell} ipython3
clean = (muons["dx"]*0.3 > 3.5) & (muons["dx"]*0.3 < 50) #& (cells['pca_length']*0.3 > 150)
#clean = np.ones(shape=muons["residual_range"].shape, dtype=np.bool)
plt.hist2d(muons["residual_range"][clean]*0.3, muons["dQ"][clean]/muons["dx"][clean],
          range=[[0, 200], [500, 4000]], bins=[40, 80],
          cmap="viridis")
plt.colorbar()
plt.xlabel("Residual range (cm)")
plt.ylabel("Muon dQ/dx [arbitrary units / cm]")
```
