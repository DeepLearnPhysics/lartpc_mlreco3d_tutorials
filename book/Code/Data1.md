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

# Using `lartpc_mlreco3d`
Before starting anything, it is good practice to look at your dataset in an event display. This chapter is strictly about the I/O part of `lartpc_mlreco3d` (independently of everything else, models, etc) and how to use it to visualize your dataset.

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


## Configuration
You need to specify a *configuration*, in YAML syntax, which tells `lartpc_mlreco3d` how you want to access the data: how many images (`batch_size`), the path to your dataset, which quantities you want to retrieve from the dataset. You can even limit the I/O to a specific list of entry numbers using `event_list`.

```{code-cell}
cfg = """
iotool:
  batch_size: 32
  shuffle: False
  num_workers: 4
  collate_fn: CollateSparse
  dataset:
    name: LArCVDataset
    data_keys:
      - DATA_DIR/wire_mpvmpr_2020_04_test_small.root
    limit_num_files: 10
    #event_list: '[6436, 562, 3802, 6175, 15256]'
    schema:
      input_data:
        parser: parse_sparse3d
        args:
          sparse_event_list: [sparse3d_reco, sparse3d_reco_chi2]
      segment_label:
        parser: parse_sparse3d
        args:
          sparse_event_list: [sparse3d_pcluster_semantics_ghost]
      cluster_label:
        parser: parse_cluster3d
        args:
          cluster_event: cluster3d_pcluster
          particle_event: particle_pcluster
          particle_mpv_event: particle_mpv
          sparse_semantics_event: sparse3d_pcluster_semantics
          add_particle_info: True
          clean_data: True
      particles_label:
        parser: parse_particles
        args:
          particle_event: particle_corrected
          cluster_event: sparse3d_pcluster
""".replace('DATA_DIR', DATA_DIR)
```

Now that the configuration is defined, you can feed it to `lartpc_mlreco3d`:

```{code-cell}
cfg=yaml.load(cfg,Loader=yaml.Loader)
# pre-process configuration (checks + certain non-specified default settings)
process_config(cfg)
# prepare function configures necessary "handlers"
hs=prepare(cfg)
```

## Iterate
One way to iterate through the dataset is using `next`:
```{code-cell}
data = next(hs.data_io_iter)
```
You can see that `data` is a dictionary whose keys match the names specified in the configuration block above:

```{code-cell}
data.keys()
```

## Taking a look at the images

It is time to run this configuration and see what we get out of it! We use Plotly to visualize the 3D images.

```{code-cell}
entry = 0
```
Let's select the first entry.

```{code-cell}
clust_label = data['cluster_label'][data['cluster_label'][:, 3] == entry]
input_data = data['input_data'][data['input_data'][:, 3] == entry]
segment_label = data['segment_label'][data['segment_label'][:, 3] == entry, -1]
particles_label = data['particles_label'][data['particles_label'][:, 3] == entry]
```

### Input data
Let us visualize the `input_data` first:

These are the energy deposits detected by the LArTPC. The energy scale might be true if you are looking at true labels,
or it might be a reconstructed energy, depending on what you are loading into `input_data`.

```{code-cell}
trace = []

trace+= scatter_points(input_data,markersize=1,color=input_data[:, -2], cmin=0,cmax=500)
trace[-1].name = 'input_data'
trace[-1].marker.colorscale='viridis'

trace+= scatter_points(input_data[segment_label < 5],markersize=1,color=input_data[segment_label < 5, -2], cmin=0,cmax=500)
trace[-1].name = 'input_data (true noghost)'
trace[-1].marker.colorscale='viridis'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Semantic labels
Let us look at the semantic labels next: for each voxel, there is a class label which takes integer values in `[0, 5]`.

```{code-cell}
trace = []

trace+= scatter_points(input_data,markersize=1,color=segment_label, cmin=0,cmax=5, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Semantic segmentation labels (including ghost points)'
trace[-1].marker.colorscale='viridis'

trace+= scatter_points(input_data[segment_label < 5],markersize=1,color=segment_label[segment_label < 5], cmin=0,cmax=5, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Semantic segmentation labels (w/o ghost points)'
trace[-1].marker.colorscale='viridis'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Points of interest
Finally, `particles_label` holds the coordinates of points of interest. They are displayed as big dots in the visualization.
The dots color corresponds to the true semantic class attributed to each point of interest.

```{code-cell}
trace = []

trace+= scatter_points(input_data[segment_label < 5],markersize=1,color=segment_label[segment_label < 5], cmin=0,cmax=5, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Semantic segmentation labels (w/o ghost points)'
trace[-1].marker.colorscale='viridis'

trace += scatter_points(particles_label, markersize=5, color=particles_label[:, 4], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = "True point labels"

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Particle instances
A particle instance is a cluster of voxels that belong to an individual particle. Each color here represents
a different particle instance.

```{code-cell}
trace = []

trace+= scatter_points(clust_label[:, :3],markersize=1,color=clust_label[:, 6], cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True cluster labels'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Interaction groups
Particle instances can then be grouped into *interactions*. Each color is a different
interaction in this visualization.


```{code-cell}
trace = []

trace+= scatter_points(clust_label[:, :3],markersize=1,color=clust_label[:, 7], cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True interaction labels'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

### Neutrino vs cosmics
There are two types of interactions: the ones due to a neutrino traversing the
detector volume (i.e. our signal!), and the ones due to cosmic rays (background).

```{code-cell}
trace = []

trace+= scatter_points(clust_label[:, :3],markersize=1,color=clust_label[:, 8], cmin=0, cmax=50, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'Nu / cosmic labels'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

## More about the I/O configuration
Keep reading if you want to understand more about that YAML configuration block.

`lartpc_mlreco3d` expects a ROOT file (created with LArCV, a C++ library to process
LArTPC images) as input file. You can explore this file with ROOT alone, if you know
how to use ROOT, but the most intuitive way to visualize it is to use the I/O module
of `lartpc_mlreco3d`.

### What are parsers?
Images are stored in ROOT + LArCV format. Parsers are functions in `lartpc_mlreco3d`
that read the information stored in the ROOT file, select just what we are interested
in (e.g. particle information ? energy depositions ?) and format it in a friendly way
for our chain.

They only need to know the names of certain quantities stored in the ROOT file (as TTree,
if you know what this is).

### I/O Configuration - a brief gist
`lartpc_mlreco3d` uses the YAML format for its configuration. Here is a skeleton config
that shows only the parameters that will matter the most for you:
```
iotool:
  batch_size: 16
  (...)
  sampler:
    name: RandomSequenceSampler
  (...)
  dataset:
    (...)
    data_keys:
      - DATA_DIR/wire_mpvmpr_2020_04_test_small.root
    (...)
    schema:
      input_data:
        - parse_sparse3d_scn
        - sparse3d_pcluster
      (...)
```

The main things to pay attention to in this data I/O configuration block are:
* batch size
* randomization (default: none if sampler is commented out, enabled if you include the RandomSequenceSampler)
* dataset filename
* the *schema*, or list of parsers and their individual configurations

The schema has a simple format: it is a list where each item is formatted as follows:

```
	my_custom_data_name:
	  - parse_whatever # this is the parser name
	  - sparse3d_pcluster # this string will be the parser's first argument
	  - sparse3d_reco # this string will be the parser's second argument
	  - # etc
```

The parser's arguments are the names of quantities stored in the input ROOT file.

 A real life example would look like this:

```
      input_data:
        - parse_sparse3d_scn
        - sparse3d_pcluster
```

This tells us that we want a field called `input_data` (our choice) in the input data
dictionary. For the sake of example let's call this input dictionary `data_blob`.
The parser name needs to be the first in the list that follows. Hence,
`data_blob['input_data']` will be the output of the parser `parse_sparse3d_scn`.
That parser will receive as arguments whatever names follow in the list - here, there
is just one, `sparse3d_pcluster`.

You can have as many such items in the schema list - each of them will be available
in the input data dictionary under the name that you specify.

Note: some stages of the chain might expect a specific name in the input dictionary.
Sometimes this is configurable in the network configuration.

### I/O Configuration - a real example
Let us show a real I/O configuration example:

```
iotool:
  batch_size: 16
  shuffle: False
  num_workers: 8
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
      - ./wire_mpvmpr_2020_04_test_small.root
    limit_num_files: 10
    schema:
      input_data:
        - parse_sparse3d_scn
        - sparse3d_pcluster
      segment_label:
        - parse_sparse3d_scn
        - sparse3d_pcluster_semantics
      cluster_label:
        - parse_cluster3d_clean_full
        - cluster3d_pcluster
        - particle_pcluster
        - sparse3d_pcluster_semantics
      particles_label:
        - parse_particle_points
        - sparse3d_pcluster
        - particle_corrected
```

These are typical inputs that you would be looking at:
* `parse_sparse3d_scn` gets you the actual 3D image, the voxels and various values for each voxel. For each voxel it will parse as many features as you are providing branch names. Here each voxel in `input_data` will have a single feature coming from `sparse3d_pcluster` (the true energy depositions). Each voxel in `segment_label` will have a single feature coming from `sparse3d_pcluster_semantics` (which holds the true semantic labels).
* `parse_cluster3d_clean_full` is parsing particle-wise information and aggregating it. For each non-zero voxel it will provide you with energy deposit, cluster id, interaction id, neutrino vs cosmic label, etc.
* `parse_particle_points` retrieves the coordinates and semantic class of points of interest for PPN.

---

Now you should be all set to browse through the images in a dataset
and loop over them.
