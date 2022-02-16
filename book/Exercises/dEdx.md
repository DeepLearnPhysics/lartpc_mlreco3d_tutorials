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
---

# Exercise 2: Muon stopping power¶

The *stopping power* of a particle usually refers to the energy loss rate $dE/dx$ when it passes through matter. When charged particles travel through our LArTPC detector, they interact with the argon and lose energy.

### MIP and Bragg peak¶

Minimally ionizing particles (MIP) are charged particles which lose energy when passing through matter at a rate close to minimal. Particles such as muons often have energy losses close to the MIP level and are treated in practice as MIP. The only exception is when the muon comes to a stop and experiences a Bragg peak.

## I. Motivation

We know that the energy loss rate of a MIP in argon is ~2 MeV/cm. Hence our goal is to carefully isolate the MIP-like sections of muons (i.e. leaving out the ends of the track), and compute the (reconstructed) energy loss along these trajectories $dQ/dx$. This can inform the detector calibration effort, for example, since we can compare the peak of the $dQ/dx$ histogram with the theoretical expected value (although there are many detector effects that make this not straightforward). We can also study the spatial uniformity of the detector by looking at MIP $dQ/dx$ values in different regions of the detector, etc.

## 2. Setup

Again, we start by setting our working environment. Some necessary boilerplate code:

+++

#### a. Software and data directory

```{code-cell} ipython3
:tags: []

import os, sys, yaml
SOFTWARE_DIR = "/home/dae/Desktop/dev/slac/lartpc_mlreco3d" # YOUR PATH TO LARTPC_MLRECO3D
DATA_DIR = os.environ.get('DATA_DIR')
# Set software directory
sys.path.append(SOFTWARE_DIR)
```

+++ {"tags": []}

#### b. Numpy, Matplotlib, and Plotly for Visualization and data handling.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')


import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
```

#### c. MLRECO specific imports for model loading and configuration setup

```{code-cell} ipython3
from mlreco.main_funcs import process_config, cycle
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
import warnings
warnings.filterwarnings('ignore')
cfg_file = '/home/dae/Desktop/dev/slac/config.cfg' # YOUR PATH TO THE CONFIG FILE
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
process_config(cfg, verbose=False)
```

#### d. Initialize and load weights to model using Trainer. 

```{code-cell} ipython3
loader = loader_factory(cfg, event_list=None)
dataset = iter(cycle(loader))
Trainer = trainval(cfg)
loaded_iteration = Trainer.initialize()
```

As usual, the model is now ready to be used (check for successful weight loading). Let's do one forward iteration to retrieve a handful of events.

```{code-cell} ipython3
data, result = Trainer.forward(dataset)
```

### Step 1. Find Muons

The particles in a given image are classified into one of five particle types: photon $\gamma$, electron $e$, muon $\mu$, pion $\pi$, and proton $p$. Obtaining particles from the full chain is quite simple: we initialize the `FullChainEvaluator` for this batch of events and examine the particle composition through the `get_particles` (for true particles, `get_true_particles`) method.

```{code-cell} ipython3
from analysis.classes.ui import FullChainEvaluator
```

```{code-cell} ipython3
# Only run this cell once!
evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
print(evaluator)
```

```{code-cell} ipython3
entry = 1    # Batch ID for current sample
print(evaluator.index[entry])
```

Let's retreive the list of predicted and true particles for this image.

```{code-cell} ipython3
particles = evaluator.get_particles(entry, primaries=False)
true_particles = evaluator.get_true_particles(entry, primaries=False)
```

### 2. Confirm PID Predictions

The particle types can be easily seen and access when using the `Particle` data structure. Below is the full list of particles participaing in this image.

```{code-cell} ipython3
particles
```

We can quickly visualize the particles using the visualization helper functions:

```{code-cell} ipython3
from mlreco.visualization.plotly_layouts import white_layout, trace_particles, trace_interactions
```

```{code-cell} ipython3
def trace_particles(particles, color='id'):
    '''
    Get Scatter3d traces for a list of <Particle> instances.
    Each <Particle> will be drawn with the color specified 
    by its unique particle ID.

    Inputs:
        - particles: List[Particle]

    Returns:
        - traces: List[go.Scatter3d]
    '''
    traces = []
    for p in particles:
        c = int(getattr(p, color)) * np.ones(p.points.shape[0])
        plot = go.Scatter3d(x=p.points[:,0], 
                            y=p.points[:,1], 
                            z=p.points[:,2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=c,
                                colorscale="rainbow",
                                reversescale=True,
                                opacity=1, cmin=0, cmax=5),
                               hovertext=int(getattr(p, color)),
                       name='Particle {}'.format(p.id)
                              )
        traces.append(plot)
    return traces
```

```{code-cell} ipython3
traces = trace_particles(particles, color='pid')
traces_true = trace_particles(true_particles, color='pid')
```

The predicted particles, each color-coded according to its semantic type, will be displayed in left; the true particles on right. 

```{code-cell} ipython3
fig = make_subplots(rows=1, cols=2,
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    horizontal_spacing=0.05, vertical_spacing=0.04)
fig.add_traces(traces, rows=[1] * len(traces), cols=[1] * len(traces))
fig.add_traces(traces_true, rows=[1] * len(traces_true), cols=[2] * len(traces_true))
fig.layout = white_layout()
fig.update_layout(showlegend=False,
                  legend=dict(xanchor="left"),
                 autosize=True,
                 height=600,
                 width=1500,
                 margin=dict(r=20, l=20, b=20, t=20))
iplot(fig)
```

```{code-cell} ipython3
muons = [p for p in particles if p.pid == 2]
```

```{code-cell} ipython3
muons
```

### Step 3. Muon direction and Binning

+++

This step is more of a Scipy/Numpy exercise! Use PCA to compute the general direction of the muon track, and bin the voxels (`np.digitize`) along this track to define segments of a certain length.

For each segment identified, sum the reconstructed energy deposited (dQ) and compute the segment length (dx). You can add some cuts such as minimal number of voxels in a segment to remove problematic segments (e.g. if there are any gaps in the track, etc).

```{code-cell} ipython3
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
edge_threshold = 12
segment_length = 5
min_voxels = 5
```

```{code-cell} ipython3
def compute_df(particles):
    
    segment_dQ = []
    segment_dx = []
    segment_dN = []

    for mp in particles:
        proj = pca.fit_transform(mp.points)

        # Cut edges
        main_body = (proj[:, 0] > proj[:, 0].min() + edge_threshold) \
                  & (proj[:, 0] < proj[:, 0].max() - edge_threshold)
        proj = proj[main_body]
        
        if proj.shape[0] <= 0:
            continue

        # Bin track into segments
        bins = np.arange(np.floor(proj[:, 0].min()), 
                         np.ceil(proj[:, 0].max()), segment_length)
        idx = np.digitize(proj[:, 0], bins)

        for i in np.unique(idx):
            segment_idx = idx == i
            dQ = mp.depositions[main_body][segment_idx].sum()
            dx = proj[segment_idx, 0].max() - proj[segment_idx, 0].min()
            dN = np.count_nonzero(segment_idx)
            if dx > 0 and dN > min_voxels:
                segment_dQ.append(dQ)
                segment_dx.append(dx)
                segment_dN.append(dN)

    segment_dQ = np.array(segment_dQ)
    segment_dx = np.array(segment_dx)
    segment_dN = np.array(segment_dN)
    return segment_dQ, segment_dx, segment_dN
```

The `compute_df` function will allow us to calculate $dQ$ and $dx$ for a given set of particles. We now iterate over all entries in the batch and collect $dQ/dx$ for plotting

```{code-cell} ipython3
from pprint import pprint
```

```{code-cell} ipython3
dQ_collect, dx_collect = [], []
for i, entry in enumerate(evaluator.index):
    particles = evaluator.get_particles(entry, primaries=False)
    muons = [p for p in particles if p.pid == 2]
    dQ, dx, _ = compute_df(muons)
    dQ_collect.append(dQ)
    dx_collect.append(dx)
dQ_collect = np.hstack(dQ_collect)
dx_collect = np.hstack(dx_collect)
```

### Step 4. Plot dQ/dx Histogram

Make a histogram with all the segments identified. Hopefully, we can see a nice peak shape:

```{code-cell} ipython3
plt.hist(dQ / dx, histtype='step', bins=20, range=[0, 5000])
plt.xlabel('dQ/dx (reconstructed charge / voxels distance)')
plt.ylabel('(Predicted) Muon Track Segments')
```

## Optional: display the PCA !

How well does the PCA reflect the muon track direction? Is our segment length reasonable w.r.t the track thickness? If you want to study this in more details, looking at some visualizations can be helpful to get a first intuition.


```{code-cell} ipython3

```
