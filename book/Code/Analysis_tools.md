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

# Introduction to `lartpc_mlreco3d` Analysis Tools

This notebook is a tutorial on how to use trained models in `lartpc_mlreco3d` for high level physics analysis, including but not limited to event selection. The purpose of this tutorial is to make inference and physics analysis using neural network models of `lartpc_mlreco3d` more accessible to non-experts and to make the process of asking simple questions (ex. how many particles in given image?) independent from knowing implementation details of separate sub-modules (which may include specific settings on hyperparameters and post-processing scripts). The tutorial is designed as follows:

 * Overview of `lartpc_mlreco3d.analysis` (analysis helper functions and classes)
 * `Particle` and `Interaction` Data Structures
 * How to use `FullChainEvaluator`
 * Example usage of `FullChainEvaluator` and how to run inference and save information via `analysis/run.py`.
 
Please send questions and any bug reports to Dae Heun (I'm best available via slack)

+++

## 1. Overview of `lartpc_mlreco3d.analysis`

[comment]: # (Why we need analysis)

The ML full chain (`lartpc_mlreco3d/mlreco/models/full_chain.py`) is a combination of multiple (convolutional and graphical) neural networks, chained together to build information from the bottom-up direction while extracting useful features on each level of detail. However, in many cases the level of detail provided by both the full chain code and its output is too excessive for asking simple high-level questions, such as: 

 * How many interactions are there in this image? 
 * How many leptons does this interaction contain? 
 * What is the five-particle classification accuracy for this model?
 * What is the signal (however one defines it) selection efficiency/purity?
 
Although it is possible to answer these questions using the output dictionary of the full chain, the pathway to that often involves asking people around how to set hyperparameters, how to call some specific post-processing functions, what convention to use when selecting viable particle/interaction candidates, etc.

The `lartpc_mlreco3d` analysis tools are designed to organize full chain output information into more human-friendly format and provide an common and convenient user interface to perform high level analysis without requiring explicit knowledge of the full chain submodules. The tools are stored under `lartpc_mlreco3d/analysis`, which is outside of all neural-network related works in `lartpc_mlreco3d/mlreco` (on purpose). The practical usage of the analysis code will be much more transparent through demonstration using the full chain. 

NOTE: Please use the latest `develop` branch from the main repository `DeepLearningPhysics/lartpc_mlreco3d/develop`. We will first load the full chain model to this notebook:

```{code-cell} ipython3
import numpy as np
import plotly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pprint import pprint

import seaborn as sns
sns.set(rc={
    'figure.figsize':(20, 10), 
})
sns.set_context('talk') # or paper

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
```

```{code-cell} ipython3
LARTPC_MLRECO_PATH = "/sdf/group/neutrino/koh0207/lartpc_mlreco3d/"    # Replace it with your copy of lartpc_mlreco3d
```

```{code-cell} ipython3
import sys, os, re
sys.path.append(LARTPC_MLRECO_PATH)   
import torch
print(torch.cuda.is_available())

from mlreco.main_funcs import process_config, train, inference, make_directories, cycle
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
import yaml
```

The full chain is trained on two datasets (as of 02/02/2022):
 * `/sdf/group/neutrino/data/mpvmpr_2020_01_v04/test.root`: MPVMPR_V04 dataset with no ghost points
   - latest config file: `/sdf/group/neutrino/koh0207/logs/mpvmpr_chain.cfg`
   - latest weights: `/sdf/group/neutrino/koh0207/weights/chain/full_chain_test_weights.ckpt`
 * `/sdf/group/neutrino/ldomine/mpvmpr_082021/test.root`: ICARUS dataset with ghost points. 
   - latest config file: `/sdf/group/neutrino/koh0207/logs/icarus.cfg`
   - latest weights: `/sdf/group/neutrino/koh0207/weights/chain/icarus_weights.ckpt`
   
Since the dataset *with* ghost points are more representative of the full chain behavior, we will use the ICARUS dataset coupled with its latest weight. The following cell loads the full chain configuration file as a dictionary (`cfg`) and builds the full chain neural network model:

```{code-cell} ipython3
cfg_file = '/sdf/group/neutrino/koh0207/logs/icarus.cfg'
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
process_config(cfg, verbose=False)
loader = loader_factory(cfg)
dataset = iter(cycle(loader))
Trainer = trainval(cfg)
loaded_iteration = Trainer.initialize()
#make_directories(inference_cfg, loaded_iteration)
```

If everything checks out, it should look like:
> Ghost Masking is enabled for UResNet Segmentation
>
> Ghost Masking is enabled for MinkPPN.
>
> Restoring weights for  from /sdf/group/neutrino/koh0207/weights/chain/icarus_weights.ckpt...
>
> Done.

Now that the full chain and its trained weights are loaded into the notebook, let's do a single forward pass and inspect the results.

```{code-cell} ipython3
data_blob, res = Trainer.forward(dataset)
```

```{code-cell} ipython3
print("Number of fieldnames in full chain output dictionary: {}".format(len(res.keys())))
```

We first introduce the `FullChainEvaluator`, which is an user-interface wrapper class for `lartpc_mlreco3d/analysis`. 

```{code-cell} ipython3
from analysis.classes.ui import FullChainEvaluator
```

To configure an instance of `FullChainEvaluator`, we need the following parameters.

 1. `data_blob` (dict of lists): dictionary containing lists of `np.ndarrays`, with list length equal to the batch size (**unwrapped**)
 2. `res` (dict): output dictionary from full chain model (**unwrapped**) 
 3. `cfg` (dict): inference `.cfg` configuration file. Mainly for setting batch size, dataset path, output log file path, etc.
 4. `deghosting` (bool, optional): whether the full chain of interest supports deghosting. This option is mostly reserved for development, and for most circumstances it will most certainly be set to `True`. 
 
 By **unwrapped**, we mean that in the inference configuration file, the `unwrapper` option should be set to `True`:
```json
trainval:
  seed: 123
  unwrapper: unwrap_3d_mink
```

```{code-cell} ipython3
predictor = FullChainEvaluator(data_blob, res, cfg, deghosting=True)   # Only run this cell once! This is due to deghosting being an in-place operation. 
```

We can check how many images are in this batch:

```{code-cell} ipython3
print(predictor)
```

## 2. Obtaining True and Predicted Labels for Visualization

Here we plot the predicted and true labels for semantic, fragment, group, and interaction predictions side-by-side using Plotly. The plotted labels are set to group labels (particle instance labels) by default, but you are free to plot other labels provided in the following cell. 

```{code-cell} ipython3
entry = 0   # Specify the batch_id 
pred_seg = predictor.get_predicted_label(entry, 'segment_label')    # Get predicted semantic labels from batch id <entry>
true_seg = predictor.get_true_label(entry, 'segment_label')    # Get true semantic labels from batch id <entry>

pred_fragment = predictor.get_predicted_label(entry, 'fragment_label')
true_fragment = predictor.get_true_label(entry, 'fragment_label')

pred_group = predictor.get_predicted_label(entry, 'group_label')
true_group = predictor.get_true_label(entry, 'group_label')

pred_interaction = predictor.get_predicted_label(entry, 'interaction_label')
true_interaction = predictor.get_true_label(entry, 'interaction_label')

pred_pids = predictor.get_predicted_label(entry, 'pdg_label')
true_pids = predictor.get_true_label(entry, 'pdg_label')

pred_plot, true_plot = pred_group, true_group
```

Here, the full chain code and the evaluator `predictor` is handling all hyperparameters and post-processing necessary to properly produce the predicted and true labels. This should allow one to inspect the full chain output at different levels of detail for better understanding of when and how the reconstruction succeeds/fails. 

Some boilerplate for better plotly formatting:

```{code-cell} ipython3
from mlreco.visualization import scatter_points, scatter_cubes, plotly_layout3d
from mlreco.utils.metrics import unique_label
layout = plotly_layout3d()
bg_color = 'rgba(0,0,0,0)'
grid_color = 'rgba(220,220,220,100)'
layout = dict(showlegend=False,
                 autosize=True,
                 height=1000,
                 width=1000,
                 margin=dict(r=20, l=20, b=20, t=20),
                 plot_bgcolor='rgba(255,255,255,0)',
                 paper_bgcolor='rgba(255,255,255,0)',
                 scene1=dict(xaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            yaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            zaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            aspectmode='cube'),
                 scene2=dict(xaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            yaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            zaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            aspectmode='cube'),
                 scene3=dict(xaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            yaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            zaxis=dict(dict(backgroundcolor=bg_color, gridcolor=grid_color)),
                            aspectmode='cube'))
```

Prediction is shown in **left**, while true labels are shown in **right**. 

```{code-cell} ipython3
make_plots = True
if make_plots:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=('Prediction', 'Truth'), 
                        horizontal_spacing=0.05, vertical_spacing=0.04)
    fig.add_trace(go.Scatter3d(x=data_blob['cluster_label'][entry][:,1], 
                               y=data_blob['cluster_label'][entry][:,2], 
                               z=data_blob['cluster_label'][entry][:,3],
                               mode='markers',
                                marker=dict(
                                    size=2,
                                    color=pred_plot,
#                                             cmax=4, cmin=-1,
                                    colorscale="rainbow",
                                    reversescale=True,
                                    opacity=1
                                ),
                               hovertext=pred_plot,
                       name='Prediction Labels'
                              ), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=data_blob['cluster_label'][entry][:,1], 
                               y=data_blob['cluster_label'][entry][:,2], 
                               z=data_blob['cluster_label'][entry][:,3],
                               mode='markers',
                                marker=dict(
                                    size=2,
                                    color=true_plot,
#                                             cmax=4, cmin=-1,
                                    colorscale="rainbow",
                                    reversescale=True,
                                    opacity=1
                                ),
                               hovertext=true_plot,
                       name='Prediction Labels'
                              ), row=1, col=2)

    fig.layout = layout
    fig.update_layout(showlegend=True,
                      legend=dict(xanchor="left"),
                     autosize=True,
                     height=1000,
                     width=2000,
                     margin=dict(r=20, l=20, b=20, t=20))
    fig.update_layout(
        scene1 = dict(
            xaxis = dict(range=[0,768],),
                         yaxis = dict(range=[0,768],),
                         zaxis = dict(range=[0,768],),),
        scene2 = dict(
            xaxis = dict(range=[0,768],),
                         yaxis = dict(range=[0,768],),
                         zaxis = dict(range=[0,768],),),
        margin=dict(r=20, l=10, b=10, t=10))
    iplot(fig)
```

## 3. Particle and Interaction Data Structures

+++

When performing event selection analysis, it is often convenient to organize information into units of particles and interactions:
 * A `Particle` is a set of voxels that share the same **predicted** group id.
 * A `TruthParticle` is a set of voxels that share the same  **true** group id. 
 * An `Interaction` is a set of voxels that share the same **predicted** interaction id. 
 * A `TruthInteraction` is a set of voxels that share the same **true** interaction id. 
 
As such, `Particle` and `Interaction` are predicted quantities reconstructed from the ML chain, while their "true" counterparts are quantities defined using true labels. Although the two classes share many of their properties, we deliberately decide to separate any truth information from interfering with predicted quantities. 

Let's process the current image's ML reco output information into these data structures. The `get_particles` function will organize the point cloud of a selected batch id (`entry`) into a list of `Particle` instances. The `primaries=True` option will only select particles that originate from a vertex:

```{code-cell} ipython3
predicted_particles = predictor.get_particles(0, primaries=False)
```

```{code-cell} ipython3
predicted_particles = predictor.get_particles(0, primaries=True)
pprint(predicted_particles)
```

Here, the `Score` refers to the softmax probability value of the most confident prediction. For example, in the above cell, the full chain predicts with 85.44% softmax probability that the particle with `ID=0` is of class `Photon`. The `Size` refers to the total number of voxels with the given ID (the voxel count of the particle).

Note that since the particles are constructed from predicted quantities, we see some "invalid" particles with size 3, 9, and so on. 

The counterpart for true particles is the function `get_true_particles`:

```{code-cell} ipython3
true_particles = predictor.get_true_particles(0)
pprint(true_particles)
```

Note that `TruthParticle` does not contain the `Score` value, and currently it does not differ between MPV (multi-particle vertex) and MPR (multi-particle rain). This means that if prediction was to be perfect (particle clustering perfectly matches that of truth labels) and `primaries` was set to `True`, the MPR particles will be omitted from the predicted list of particles while remaining in the list of true particles. 

+++

Similarly, for interactions we have:

```{code-cell} ipython3
predicted_interactions = predictor.get_interactions(entry, primaries=True)
pprint(predicted_interactions)
```

Here, we see that `Interaction 0` has 7 particles with particle IDs `[0, 1, 3, 4, 8, 9, 10]` (not to be confused with PDG codes, the particle IDs are nominal integer values used to separate distinct instances), with a vertex located at $\vec{v}$ = (682.02, 558.37, 632.04). When performing `get_interactions`, the algorithm first retrives all particles in the given entry using `get_particles`, groups the particles into separate `Interaction` classes, and then runs a post-processing script specifically reserved for vertex prediction. 

Each entry in `predicted_interactions` has data type `Interaction`. For example, you can retrive the list of particles contained in this interactions is `List[Particle]`:

```{code-cell} ipython3
ia_0 = predicted_interactions[0]
pprint(ia_0.particles)
```

It is also straightforward to get true particles:

```{code-cell} ipython3
predicted_interactions = predictor.get_true_interactions(entry, primaries=True)
pprint(predicted_interactions)
```

More information could be read from the docstrings of `Particle/TruthParticle` and `Interaction/TruthInteraction` in `analysis/particle.py`. 

+++

## 4. Matching particles and interactions

+++

To evaluate the performance of the full chain, we require a matching procedure that relates predicted `Particles/Interactions` with the most appropriate `TruthParticles/TruthInteractions`. Several issues arise when doing this:
 1. One can either start with the set of predictions and match a corresponding true particle **for each** predicted entity (`pred -> true` matching), or vice versa (`true -> pred` matching). The result will in general be different. 
 2. For a `pred -> true` matching scheme, it is possible for two or more predicted particles to map to the same true particle (the most obvious example happens when a single true particle is fragmented into separate predictions, due to mistakes in clustering). 
 3. For a `pred -> true` matching scheme, it is possible for a given predicted particle to have no match whatsoever (ex. a fake particle is created due to mistakes in deghosting, or a particle not associated with an interaction is mistakenly assigned an interaction vertex).
 
The above points hold similarly for the reverse case (`true -> pred`). 

The matching algorithms for particles and interactions are implemented in `match_particles` and `match_interactions` member functions:

```{code-cell} ipython3
matched_particles = predictor.match_particles(entry, primaries=True, mode='pt')
pprint(matched_particles)
```

Here, `matched_particles` will be a list of (`Particle, TruthParticle`) tuples (in Python typing notation--`List[Tuple[Particle, TruthParticle]]`). We can see that particles with similar sizes are matched together. 
 * Note: if a true particle match does not exist for a given prediction, the method will place a `None` instead of assigning any `TruthParticle` instance (as could be seen above).
 * The `mode='pt'` refers to the matching convention `pred -> true`. For `true -> pred` matching, we simply replace it as `mode='tp'`.

The `match_particles` method does not take into account interaction grouping information to match particles. That is, regardless of interaction grouping it will work with predicted and true particles of a given image, and work out the best possible matching configuration for the two sets. 

If one desires to match particles hierarchically, proceeding from matching interactions and then particles within those interactinos, `match_interactions(entry, match_particles=True)` is the correct method:

```{code-cell} ipython3
matched_interactions = predictor.match_interactions(entry)    # Default mode is match_particles=True
```

```{code-cell} ipython3
matched_interactions
```

Now we see that the `Match = [...]` string is filled with the matched particle ID values. More precisely, we read the above result as follows:
 * `Particle 0` inside `Interaction 0` is matched with `TruthParticle 0` inside `TruthInteraction 5`
 * `TruthParticle 0` has 5 matches `[3, 0, 4, 1, 8]`, meaning (predicted) `Particles [3,0,4,1,8]` all have the maximum overlap with the same `TruthParticle 0`. 

+++

## 5. Example: Evaluating full chain five-particle classification performance using `lartpc_mlreco3d/analysis`.

+++

Let's say you want to produce a confusion matrix to evaluate five-particle classification performance on the full chain, using the `icarus` ghost-inclusive dataset. You first start by defining a selection algorithm under `analysis/algorithms/selection.py`. Let's call the algorithm name "five_particle_classification". The decorator function under `analysis.decorator` is designed to handle most of the common boilerplate code such as importing the config file, initializing the `Trainval` class, iterating the dataset for a given number of iterations, and saving the results to a csv file (many of the code was borrowed from the same design principle used in `mlreco/post_processing`). The specific details of your analysis script (in this case, saving particle type information) is independent from this boilerplate. So let's define a function `five_particle_classification` which will match interactions, match particles, and retrive predicted and true particle type information, which we intend to save as a csv file. 

```python
from collections import OrderedDict
from analysis.classes.ui import FullChainEvaluator

from analysis.decorator import evaluate
from analysis.classes.particle import match

@evaluate(['result_pt.csv', 'result_tp.csv'], mode='per_batch')
def five_particle_classification(data_blob, res, data_idx, analysis_cfg, module_config):
    
    pred_to_truth, truth_to_pred = [], []
    
    return [pred_to_truth, truth_to_pred]
```

+++

The first argument of the `evaluate` decorator is a list of strings indicating the names of the output csv files. Here, I want to save my results on two separate files, one for `pred -> true` matching (`result_pt.csv`) and another for `true -> pred` matching (`result_tp.csv`). The outputs `pred_to_truth` and `truth_to_pred` are each list of dictionaries, where each dictionary represents a single row in the output csv file, with keys indicating the column headers. 

+++

### A. General form of the evaluation function.

As one can observe from the above example, an evaluation function takes the following form:
```python
@evaluate([filename_1, filename_2, ..., filename_n], mode=MODE_STRING)
def eval_func(data_blob, res, data_idx, analysis_cfg, cfg):
    
    ...
    
    return [output_1, output_2, ..., output_n]
```
Every evaluation function `eval_func` should have five input parameters as shown above, and must be accompanied with the `@evaluate` decorator. Here, `data_blob` and `res` are in exactly the same format as we used before in this notebook: the former refers to the input data dictionary while the latter refers to the output dictionary from the full chain. The parameter `data_idx` is a capture variable from the `evaluate` decorator, indicating the batch id number of a given image when running the `eval_func` on per single image mode. For the purpose of this tutorial we can ignore this. The important parameters are the following:
 * `analysis_cfg`: configuration dictionary for running analysis jobs. 
 * `cfg`: the config file for the full chain, as was used when initializing `FullChainEvaluator`. 

This is best understood with an example: since we defined the function `five_particle_classification` under `analysis/algorithms/selection.py`, we make a file `analysis.cfg`:

#### analysis.cfg
```json
analysis:
  name: five_particle_classification
  ...
```

We know that the `FullChainEvaluator` must run under the `deghosting` mode turned on, which is a parameter for the `FullChainEvaluator`. So we define that in the `analysis.cfg` file:

#### analysis.cfg
```json
analysis:
  name: five_particle_classification
  deghosting: True
  ...
```

+++

### B. Defining fieldnames and default values.

+++

For the csv logger to work consistently, we must specify the name of the columns and the default values for each of the columns, in case an unmatched particle or interaction pair occurs. These go under the name `fields`. We also specify the output log directory in which the output csv file will be stored. 

The decorator also has an option `mode='per_batch'`, which means that the `five_particle_classification` function will be executed for a given batch rather than on a single image. This means that `data_blob` will contain `BATCH_NUMBER` units of data, which we can iterate over to access single images. The choice `mode='per_batch'` is mostly a matter of convenience, as `FullChainEvaluator` can already handle batches as we seen before. 

```python
@evaluate(['result_pt.csv', 'result_tp.csv'], mode='per_batch')
def five_particle_classification(data_blob, res, data_idx, analysis_cfg, module_config):
    
    # Set default fieldnames and values. (Needed for logger to work)
    fields = OrderedDict(analysis_cfg['analysis']['fields'])
    pred_to_truth, truth_to_pred = [], []
    deghosting = analysis_cfg['analysis']['deghosting']
    
    pred_to_truth, truth_to_pred = [], []
    
    return [pred_to_truth, truth_to_pred]
```

+++

#### analysis.cfg
```json
analysis:
  name: five_particle_classification
  processor_cfg:
    spatial_size: 768
  log_dir: /sdf/group/neutrino/koh0207/lartpc_mlreco3d/logs/icarus
  iteration: 400
  deghosting: True
  fields:
    index: -1
    pred_particle_type: -1
    true_particle_type: -1
  ...
```
Here we've placed nominal values of `-1` for default particle types. 

+++

```python
@evaluate(['result_pt.csv', 'result_tp.csv'], mode='per_batch')
def five_particle_classification(data_blob, res, data_idx, analysis_cfg, module_config):
    # Set default fieldnames and values. (Needed for logger to work)
    fields = OrderedDict(analysis_cfg['analysis']['fields'])
    pred_to_truth, truth_to_pred = [], []
    deghosting = analysis_cfg['analysis']['deghosting']

    predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    image_idxs = data_blob['index']

    # 1. Get Pred -> True matching (for each predicted particle, match one of truth)
    for i, index in enumerate(image_idxs):

        matches = predictor.match_interactions(i, mode='pt', match_particles=True)
        
        for interaction_pair in matches:
            pred_int, true_int = interaction_pair[0], interaction_pair[1]

            if true_int is not None:
                pred_particles, true_particles = pred_int.particles, true_int.particles
            else:
                pred_particles, true_particles = pred_int.particles, []

            matched_particles, _, _ = match(pred_particles, true_particles, 
                                            primaries=True, min_overlap_count=1)
            for ppair in matched_particles:
                update_dict = OrderedDict(fields) # Initialize the csv row update dictionary
                update_dict['index'] = index    # In case we want to save image numbers for mistakes analysis. 
                if ppair[1] is not None:   # Check for unmatched pairs (particle, None)
                    update_dict.update(
                        {
                            'pred_particle_type': ppair[0].pid,   # pid means particle type
                            'true_particle_type': ppair[1].pid
                        }
                    )
                    pred_to_truth.append(update_dict)
                    
    ...
    # (do the same with true -> pred)
    
    return [pred_to_truth, truth_to_pred]
```

+++

Once this is done, you call `analysis/run.py` from the terminal as follows:

+++

```
$ python3 $PATH_TO_LARTPC_MLRECO3D/analysis/run.py $PATH_TO_CFG $PATH_TO_ANALYSIS_CFG
```

```{code-cell} ipython3
pwd
```

```{code-cell} ipython3
cp /sdf/home/k/koh0207/analysis/Event_t
```
