# lartpc_mlreco3d Tutorials

This is a collection of Jupyter notebooks and tutorials to get you up to speed on [`lartpc_mlreco3d`](https://github.com/DeepLearnPhysics/lartpc_mlreco3d).

## Install
You do not need to install these packages in order to contribute to the repository (see below how to contribute). However if you wish to build and/or push to Github Pages, these will be useful:

```bash
$ pip install -U jupyter-book ghp-import
```

## Contributing

Create your Markdown files / Jupyter notebooks. Reference them in the TOC (`_toc.yml`).

Build:

```bash
$ jupyter-book build lartpc_mlreco3d_tutorials/
```

You can now open the file `_build/html/index.html` to preview your changes.

## Updating Github Pages

If you have the right access permissions and the package `ghp-import` installed, you can update the Github Pages after building:

```bash
$ ghp-import -n -p -f lartpc_mlreco3d_tutorials/_build/html
```

## Built with
Using the awesome [Jupyter Book](https://jupyterbook.org/) and [Binder](https://mybinder.readthedocs.io/). Hosted on Github Pages.