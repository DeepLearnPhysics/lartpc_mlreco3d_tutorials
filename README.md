# lartpc_mlreco3d Tutorials

This is a collection of Jupyter notebooks and tutorials to get you up to speed on [`lartpc_mlreco3d`](https://github.com/DeepLearnPhysics/lartpc_mlreco3d).

## Install
Install Jupyter Book to contribute to the repository (see below how to contribute). 

```bash
$ pip install -U jupyter-book 
```

## Contributing

Create your Markdown files / Jupyter notebooks. Reference them in the TOC (`_toc.yml`). 
See https://jupyterbook.org/intro.html for more guidance on how to write your pages.

The first time that you build, you need to run `sh setup.sh` in the root of the repository in order to copy the necessary weight files/datasets. This currently requires you to have access to the SDF filesystem where they are stored and then copied to the `Code/` folder.

Then every time you want to build:

```bash
$ jupyter-book build book
```

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
