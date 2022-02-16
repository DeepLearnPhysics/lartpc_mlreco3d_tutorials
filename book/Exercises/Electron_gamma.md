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

# Exercise 3: electron vs gamma separation

## 1. Introduction

Electrons are visible in a LArTPC detector because of the electromagnetic showers that they trigger.

Photons, on the other hand, are neutral (no charge) and thus remain invisible to the LArTPC eyes until they convert into electrons (pair production) or Compton scatter. In both cases, the visible outcome will be an electromagnetic shower.

How can we differentiate the two, then? The answer is in the very beginning of the EM shower. For an electron, this shower will be topologically connected to the interaction vertex where the electron was produced. For a photon, there will be a gap (equal to the photon travel path) until the EM shower start (when the photon becomes indirectly visible through pair production or Compton scatter). That seems simple enough, right? Wrong, of course.

Energetic photons could interact at a distance short enough from the interaction vertex, that we would not be able to see the gap. Or, the hadronic activity might be invisible, because it includes neutral particles or because the particles are too low energy to be seen. In that case the interaction vertex might be hard to identify, and the notion of a gap goes away too. For such cases, fortunately, there is another way to tell electrons from gamma showers. Another major difference is in the energy loss rate at the start of the EM shower. An electron would leave ionization corresponding to a single ionizing particle, whereas a pair of electron + positron coming from a photon pair production would add up to two ionizing particle. Thus, we expect the dE/dx at the beginning of the shower to be roughly twice larger in the case of a gamma-induced shower compared to an electron-induced shower.

Why do we care? The difference becomes significant if, for example, you are looking for electron neutrinos. One of the key signatures you would be looking for are electrons.

In this exercise, we will focus on finding the start of EM showers and computing the reconstructed dQ/dx in these segments. Optionally, you could compare that to the result of using automatic PID as predicted by the chain.

+++

## 2. Setup

+++

### a. Software and data directory

```{code-cell} ipython3
import os, sys, yaml
SOFTWARE_DIR = "/home/dae/sdf_home/lartpc_mlreco3d" # YOUR PATH TO LARTPC_MLRECO3D
DATA_DIR = os.environ.get('DATA_DIR')
# Set software directory
sys.path.append(SOFTWARE_DIR)
```

### b. Numpy, Matplotlib, and Plotly for Visualization and data handling.

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

### c. MLRECO specific imports for model loading and configuration setup

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

### d. Initialize and load weights to model using Trainer. 

```{code-cell} ipython3
loader = loader_factory(cfg, event_list=None)
dataset = iter(cycle(loader))
Trainer = trainval(cfg)
loaded_iteration = Trainer.initialize()
```

Let's load one iteration worth of data into our notebook:

```{code-cell} ipython3
data, result = Trainer.forward(dataset)
```

## e. Setup Evaluator

```{code-cell} ipython3
from analysis.classes.ui import FullChainEvaluator
```

```{code-cell} ipython3
# Only run this cell once!
evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
print(evaluator)
```

```{code-cell} ipython3
entry = 4    # Batch ID for current sample
print("Batch ID = ", evaluator.index[entry])
```

## 3. Identifying Shower Primaries
### Step 1: Get shower primary fragments

+++

By using the `primaries=True` option, we can select out primary particles in this image. We will also load `true_particles` for comparison.

```{code-cell} ipython3
particles = evaluator.get_particles(entry, primaries=True)
true_particles = evaluator.get_true_particles(entry, primaries=True)
```

```{code-cell} ipython3
from pprint import pprint
pprint(particles)
```

Alternatively, as you may have noticed, the primariness information is also stored in the `Particle` instance as an attribute with name `is_primary`. If you prefer to view the full image and then select out primaries manually:

```{code-cell} ipython3
particles = evaluator.get_particles(entry, primaries=False)
true_particles = evaluator.get_true_particles(entry, primaries=False)
```

```{code-cell} ipython3
from pprint import pprint
pprint(particles)
```

Let's quickly plot the particles and visualize which ones are predicted as primaries. Here is one way to do it with the `trace_particles` function:

```{code-cell} ipython3
from mlreco.visualization.plotly_layouts import white_layout, trace_particles, trace_interactions
```

```{code-cell} ipython3
traces = trace_particles(particles, color='is_primary', colorscale='rdylgn')   # is_primary for coloring with respect to primary label
traces_true = trace_particles(true_particles, color='is_primary', colorscale='rdylgn')
```

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

The green voxels are predicted primary particles, while red indicates non-primary. 

It is often easier to further break down the shower into different fragments and locate which of the shower fragments actually correspond to a predicted primary.

```{code-cell} ipython3
fragments = evaluator.get_fragments(entry)
```

```{code-cell} ipython3
traces = trace_particles(fragments, color='is_primary', colorscale='rdylgn')   # is_primary for coloring with respect to primary label
traces_right = trace_particles(fragments, color='id', colorscale='rainbow')   # This time, we'll plot the predicted particle 
```

```{code-cell} ipython3
fig = make_subplots(rows=1, cols=2,
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    horizontal_spacing=0.05, vertical_spacing=0.04)
fig.add_traces(traces, rows=[1] * len(traces), cols=[1] * len(traces))
fig.add_traces(traces_right, rows=[1] * len(traces_right), cols=[2] * len(traces_right))
fig.layout = white_layout()
fig.update_layout(showlegend=False,
                  legend=dict(xanchor="left"),
                 autosize=True,
                 height=600,
                 width=1500,
                 margin=dict(r=20, l=20, b=20, t=20))
iplot(fig)

# TODO: Plot true fragment labels
```

### Step 2: Identify the startpoint of the shower primary

+++

During initialization of the `Particle` instance, PPN predictions are assigned to each particle if the distance between then is less than a predetermined threshold (`attaching_threshold`). PPN predictions that are matched to particles in this way are then stored in each `Particle` instance as attributes (`ppn_candidates`)

```{code-cell} ipython3
print("Minimum voxel distance required to assign ppn prediction to particle fragment = ", evaluator.attaching_threshold)
```

```{code-cell} ipython3
fragments = evaluator.get_fragments(entry, primaries=False)
```

The first three columns are the $(x,y,z)$ coordinates of the PPN points. The fourth column is the PPN prediction score, and the last column indicates the predicted semantic type of the point. 

We first visualize whether the predicted ppn candidates accurately locate the shower fragment start:

```{code-cell} ipython3
traces = trace_particles(fragments, color='id', size=1, scatter_ppn=True, highlight_primaries=True)   # Set scatter_ppn=True for plotting PPN information
traces_true = trace_particles(true_particles, color='id', size=1)
```

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

The left scatterplot highlighits primary shower fragments and its ppn candidates, while non-primaries are showed with faded color. The right plot shows true particle labels.

Identifying the primary shower fragments (as above) allow us to select all the voxels of the primary fragment which are close to the shower start, i.e. within some radius of the predicted PPN shower point. Of course, as expected from the scatterplot above, we may also include some cuts on the total voxel count to pick shower primary fragments that are large enough for our $dQ/dx$ analysis. 

For convenience, from now on we will only work with primary fragments:

```{code-cell} ipython3
fragments = evaluator.get_fragments(entry, primaries=True)
```

### Step 3. Compute $dQ/dx$ near the shower start

Let's first fix some parameters for our $dQ/dx$ computation. Let's say the we select all points within a radius of 10 voxels from the predicted PPN shower start point of a given primary fragment and require that the selected segment size should at least be 3 voxels long. 

```{code-cell} ipython3
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
min_segment_size = 3 # in voxels
radius = 10 # in voxels
pca = PCA(n_components=2)
```

Write a `compute_shower_dqdx` function that takes a list of primary fragments and returns a list of computed dQ/dx values for each fragment. 

```{code-cell} ipython3
def compute_shower_dqdx(frags, r=10, min_segment_size=3):
    '''
    Inputs:
        - frags (list of ParticleFragments)
        
    Returns:
        - out: list of computed dQ/dx for each fragment
    '''
    out = []
    for frag in frags:
        assert frag.is_primary  # Make sure restriction to primaries
        if (frag.startpoint < 0).any():
            continue
        ppn_prediction = frag.startpoint
        dist = cdist(frag.points, ppn_prediction.reshape(1, -1))
        mask = dist.squeeze() < r
        selected_points = frag.points[mask]
        proj = pca.fit_transform(selected_points)
        dx = proj[:, 0].max() - proj[:, 0].min()
        if dx < min_segment_size:
            continue
        dq = np.sum(frag.depositions[mask])
        out.append(dq / dx)
    return out
```

```{code-cell} ipython3
compute_shower_dqdx(fragments)
```

### Step 4. Collect data over multiple images and plot results

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
iterations = 100

collect_dqdx = []
for iteration in range(iterations):
    data, result = Trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    for entry, index in enumerate(evaluator.index):
#         print("Batch ID: {}, Index: {}".format(entry, index))
        fragments = evaluator.get_fragments(entry, primaries=True)
        dqdx = compute_shower_dqdx(fragments, r=radius, min_segment_size=min_segment_size)
        collect_dqdx.extend(dqdx)
        
collect_dqdx = np.array(collect_dqdx)
```

```{code-cell} ipython3
collect_dqdx
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

plt.hist(collect_dqdx, range=[0, 10000], bins=50)
plt.xlabel("dQ/dx")
plt.ylabel("Predicted primary shower fragments")
```

```{code-cell} ipython3

```
