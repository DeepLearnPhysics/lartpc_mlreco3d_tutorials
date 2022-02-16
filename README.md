# lartpc_mlreco3d Tutorials

This is a collection of Jupyter notebooks and tutorials to get you up to speed on [`lartpc_mlreco3d`](https://github.com/DeepLearnPhysics/lartpc_mlreco3d). You can view the tutorial page [here](https://deeplearnphysics.org/lartpc_mlreco3d_tutorials/).

## Install
Install Jupyter Book to contribute to the repository (see below how to contribute). 

```bash
$ pip install -U jupyter-book 
```

## Contributing

### Writing content
Create your Markdown files / Jupyter notebooks. Reference them in the TOC (`_toc.yml`). 
See https://jupyterbook.org/intro.html for more guidance on how to write your pages.

For better version control, it is preferred that you write your Jupyter notebook using 
Markdown. A Jupyter notebook written entirely in Markdown needs a YAML frontmatter and looks like this:

````md
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

# Your title

```{code-cell}
code that will be executed like a Jupyter notebook cell
```

````

See https://jupyterbook.org/file-types/myst-notebooks.html for more information.

### Getting the weight files and dataset to build the tutorials
The tutorials rely on some weight files and small datasets. You can download them all by running the `setup.sh` script
from the root of the repository:

```bash
$ source setup.sh (optional: path/to/your/folder)
```

Files are stored by default in the folder `lartpc_mlreco3d_tutorials/book/data`.
If you provide a custom path, the script will export that path in the environment variable `DATA_DIR`
which is used by the tutorials.

### Working on SDF filesystem (SLAC-users only)
Files will be downloaded from SDF. As such, to view the tutorials properly you need access to the SLAC SDF file system. One way to do this is to mount the remote filesystem to your local machine using [sshfs](https://help.ubuntu.com/community/SSHFS):

```bash
$ sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 $USERNAME@sdf-login.slac.stanford.edu:$PATH_TO_TUTORIAL $PATH_TO_TUTORIAL_LOCAL
```
where:
 * `$USERNAME`: your SLAC `sdf` login ID. 
 * `$PATH_TO_TUTORIAL`: path to the `lartpc_mlreco3d_tutorials` folder inside SDF.
 * `$PATH_TO_TUTORIAL_LOCAL`: your desired path to the tutorial folder, in your local machine. 

For example, after git cloning this repository in `/somewhere/lartpc_mlreco3d_tutorials`, I can make a folder `~/lartpc_mlreco3d_tutorials` in my local Ubuntu, and mount the tutorials folder in SDF to this folder as follows:
```bash
sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 USERNAME@sdf-login.slac.stanford.edu:/somewhere/lartpc_mlreco3d_tutorials ~/lartpc_mlreco3d_tutorials
```
This allows one to view files via the file browser of your local machine. 

### Building
Every time you want to build:

```bash
$ jupyter-book build book
```

### Previewing Changes

You can now open the file `_build/html/index.html` to preview your changes.

## Updating Github Pages

If you have the right access permissions and the package `ghp-import` installed:
```bash
$ pip install -U ghp-import
```

then you can easily update the Github Pages after building:

```bash
$ ghp-import -n -p -f book/_build/html
```

## Built with
Using the awesome [Jupyter Book](https://jupyterbook.org/) and [Binder](https://mybinder.readthedocs.io/). Hosted on Github Pages.
