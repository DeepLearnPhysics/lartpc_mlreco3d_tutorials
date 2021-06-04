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
---

# Using ROOT only

If you know [ROOT](https://root.cern/), the framework developed by CERN for particle physics data analysis, you can use it to look directly at the data files. 

```{note}
Ignore this if you do not know ROOT. For most purposes, you do not need to know ROOT in order to use `lartpc_mlreco3d`.
```

## In ROOT
In the command line, you can open the file with `root -b -l larcv.root`. The option `-b` (batch mode) disables graphical usage and the option `-l` tells it to skip the ROOT banner. Now you can examine it interactively using the usual ROOT commands.

The ROOT file contains several TTrees (which you can list with `.ls` in the prompt). There are three types of TTrees in our ROOT files. The type of object stored in these TTrees depends on their name:
* `sparse3d_*` refers to a [larcv::EventSparseTensor3D](https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/EventVoxel3D.h#L47).
* `cluster3d_*` refers to a [larcv::EventClusterVoxel3D](https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/EventVoxel3D.h#L27), a list of clusters of voxels.
* `particle_*` refers to a [larcv::EventParticle](https://github.com/DeepLearnPhysics/larcv2/blob/develop/larcv/core/DataFormat/EventParticle.h#L26), a list of `larcv::Particle` objects.

## In Python
Another way is to use PyROOT, the Python interface for ROOT. 

```{code-cell} 
import ROOT
from ROOT import  TFile, TChain
```

Now open the file using ROOT:
```{code-cell}
example = TFile( ' /sdf/home/l/ldomine/larcv.root' )
example.ls()
```

You can then manually browse the file. For example, if you wanted to look at the tree `cluster3d_sed_tree`, see how many clusters are there and what is the size of each cluster:

```{code-cell}
tree = example.Get("cluster3d_sed_tree")

for entry in range(tree.GetEntries()):
    tree.GetEntry(entry)
    event = tree.cluster3d_sed_branch
    clusters = event.as_vector()
    print("Number of clusters = ", len(clusters))
    for c in clusters:
        clust = c.as_vector()
        print("\t", clust.size())
```

 * [PyROOT - Getting started](https://root.cern/manual/python/#getting-started)
 
 ## Using LArCV to retrieve the image and visualize
 
 ```{note} TODO
 Coming soon
 ```