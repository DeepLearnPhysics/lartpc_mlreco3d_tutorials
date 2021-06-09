# Setting up your environment

The easiest way to work with `lartpc_mlreco3d` is to work from a container that has
all the required libraries pre-installed for you. 

```{note}
We are also working to make this work out of the box with Binder. 
```

## Using a Docker image
You can pull a Docker image directly from Docker Hub:

```bash
docker pull deeplearnphysics/larcv2:ub20.04-cuda11.0-pytorch1.7.1-extra
```

To see which images are present on your system, you can use `docker images`. It will look something like this:

```
$ docker images
REPOSITORY                TAG                             IMAGE ID            CREATED             SIZE
deeplearnphysics/larcv2   ub18.04-cuda10.2-pytorch1.7.1   4f24b6e84a1d        1 months ago        8.91GB
```

Then to run it interactively (`8edf37c1bcec` is the image ID):

```bash
docker run -i -t 8edf37c1bcec bash
```

## Cloning the `lartpc_mlreco3d` repository
Most of the notebooks in this section will assume that you have `lartpc_mlreco3d` in your `$HOME` directory.

```bash
$ git clone https://github.com/Temigo/lartpc_mlreco3d.git $HOME/lartpc_mlreco3d
$ cd $HOME/lartpc_mlreco3d
$ git checkout develop
```

## Assets
You will need a small dataset and weight files in order to run these notebooks. Here are the individual links for reference:
* [weights_full5_snapshot-999.ckpt](https://drive.google.com/file/d/1-ptcD6dHyVtxdgfo6dQLdUSrSZPlnvlz/view?usp=sharing)
* [weights_ppn3_snapshot_1999.ckpt](https://drive.google.com/file/d/155yaJ6YMEZmZBGkT8DYQijakP919dOuJ/view?usp=sharing)
* [wire_mpvmpr_2020_04_test_small.root](https://drive.google.com/file/d/1UNPtKemYkUYuLc2kGZmjKftFHKu5uXbG/view?usp=sharing)

You can download everything at once by running this `setup.sh` script and providing as first argument the path
to the folder where you want to store the files:

```bash
$ wget https://raw.githubusercontent.com/DeepLearnPhysics/lartpc_mlreco3d_tutorials/master/setup.sh
$ sh setup.sh path/to/your/folder
```.

The script will also export the environment variable `DATA_DIR` as `path/to/your/folder` for future reference in the tutorials.