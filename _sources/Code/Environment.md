# Setting up your environment

The easiest way to work with `lartpc_mlreco3d` is to work from a container that has
all the required libraries pre-installed for you.

```{note}
We are also working to make this work out of the box with Binder.
```

```{warning}
As of now (April 2022) we have an issue with one of our dependencies which **requires**
you to run ``lartpc_mlreco3d`` on a GPU for it to work as expected.
```

## Using a Docker image

```{warning}
Please make sure that you have followed Docker's post-install step that allows you to use `docker` commands without `sudo` permission. See https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user. Otherwise, you may see an error such as `permission denied while trying to connect to the Docker daemon socket` because your user does not have proper permissions to access `/var/run/docker.sock`.
```

You can pull a Docker image directly from Docker Hub:

```bash
docker pull deeplearnphysics/larcv2:ub20.04-cuda11.3-cudnn8-pytorch1.10.0-extra
```

To see which images are present on your system, you can use `docker images`. It will look something like this:

```
$ docker images
REPOSITORY                TAG                                     IMAGE ID            CREATED             SIZE
deeplearnphysics/larcv2   ub20.04-cuda11.3-cudnn8-pytorch1.10.0   4f24b6e84a1d        1 months ago        8.91GB
```

Then to run it interactively (`4f24b6e84a1d` is the image ID which you could see above):

```bash
docker run -i -t 4f24b6e84a1d bash
```

## Cloning the `lartpc_mlreco3d` repository
Most of the notebooks in this section will assume that you have `lartpc_mlreco3d` in your `$HOME` directory.

```bash
$ git clone https://github.com/DeepLearnPhysics/lartpc_mlreco3d.git $HOME/lartpc_mlreco3d
$ cd $HOME/lartpc_mlreco3d
$ git checkout v2.4.2
```

```{note}
The current stable tag is ``v2.4.2``, but feel free to update/change this as needed.
The active development branch is ``develop``.
```

## Assets
You will need a small dataset and weight files in order to run these notebooks. Here are the individual links for reference:
* [weights_full_mpvmpr_062022.ckpt](https://drive.google.com/file/d/1ZqCMHZBuBQpQYWRBuUv7dDFLvIhtHIwO/view?usp=sharing)
* [mpvmpr_062022_test_small.root](https://drive.google.com/file/d/1cO0eIXBtD8W6e6MaofT4K8w6KcAFiw5Y/view?usp=sharing)

You can download everything at once by running this `setup.sh` script and providing as first argument the path
to the folder where you want to store the files:

```bash
wget https://raw.githubusercontent.com/DeepLearnPhysics/lartpc_mlreco3d_tutorials/master/setup.sh
source setup.sh
```.


By default, this will save the files into `./book/data/`. The script will also export the environment variable `DATA_DIR` for future reference in the tutorials.

Optionally, you can provide a path to the folder where you want to store the files:


```bash
source setup.sh path/to/your/folder
```

Then `DATA_DIR` is exported as `path/to/your/folder` and the files are moved to this folder.
