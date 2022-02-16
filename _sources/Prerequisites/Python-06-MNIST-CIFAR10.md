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


# Streaming Input Dataset

We already covered the basics of an iterable dataset ([Python basics](./Pytorch-02-Python.md)) and pytorch's `DataLoader` ([pytorch introduction](Python-05-Pytorch.md)). In this notebook, we introduce a few datasets that will be used during the hands-on sessions, and practic looping over the dataset using `DataLoader`!
```{code-cell}
import torch
import numpy as np
SEED=123
np.random.seed(SEED)
torch.manual_seed(SEED)
```

## MNIST dataset

MNIST is widely used for an introductory machine learning (ML) courses/lectures. Most, if not all, ML libraries provide an easy way (API) to access MNIST and many publicly available dataset. This is true in `pytorch` as well. MNIST dataset in `Dataset` instance is available from `torchvision`. 

### Creating MNIST Dataset
A `torchvision` is a supporting module that has many image-related APIs including an interface (and management) of MNIST dataset. Let's see how we can construct:
```{code-cell}
import os
from torchvision import datasets, transforms
# Data file download directory
LOCAL_DATA_DIR = './mnist-data'
os.makedirs(LOCAL_DATA_DIR,exist_ok=True)
# Use prepared data handler from pytorch (torchvision)
dataset = datasets.MNIST(LOCAL_DATA_DIR, train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
```

Here, MNIST is also a type `Dataset` (how? through class inheritance). All torch `Dataset` instance have tow useful and common functions: the length representations and data element access via index.
```{code-cell}
print( len(dataset)  )
print( type(dataset[0]) )
```

That being said, how each data element is presented depends on a particular `Dataset` implementation. In case of MNIST, it is a tuple of length 2: **data** and **label**.
```{code-cell}
ENTRY=0
data, label = dataset[ENTRY]
print('Type of data  :', type(data),  'shape', data.shape)
print('Type of label :', type(label), 'value', label)
```

MNIST is an image of a hand-written digit in 28x28 pixels, gray scale. Note that the data `shape` is `[1,28,28]`. This follows the convention in Pytorch for image data represented as $(Cannel,Height,Width)$, or in short $(C,H,W)$. Let's visualize using `matplotlib.pyplot.imshow`. This function can take $(H,W)$ of a gray scale image. 
```{code-cell}
import matplotlib.pyplot as plt
%matplotlib inline

# Draw data
data = data.reshape(data.shape[1:])
plt.imshow(data,cmap='gray')
plt.show()
```

Let us define a function that can list images and labels in the dataset.
```{code-cell}
def plot_dataset(dataset,num_image_per_class=10):
    import numpy as np
    num_class = 0
    classes = []
    if hasattr(dataset,'classes'):
        classes=dataset.classes
        num_class=len(classes)
    else: #brute force
        for data,label in dataset:
            if label in classes: continue
            classes.append(label)
        num_class=len(classes)
    
    shape = dataset[0][0].shape
    big_image = np.zeros(shape=[3,shape[1]*num_class,shape[2]*num_image_per_class],dtype=np.float32)
    
    finish_count_per_class=[0]*num_class
    for data,label in dataset:
        if finish_count_per_class[label] >= num_image_per_class: continue
        img_ctr = finish_count_per_class[label]
        big_image[:,shape[1]*label:shape[1]*(label+1),shape[2]*img_ctr:shape[2]*(img_ctr+1)]=data
        finish_count_per_class[label] += 1
        if np.sum(finish_count_per_class) == num_class*num_image_per_class: break
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),facecolor='w')
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.imshow(np.transpose(big_image,(1,2,0)))
    for c in range(len(classes)):
        plt.text(big_image.shape[1]+shape[1]*0.5,shape[2]*(c+0.6),str(classes[c]),fontsize=16)
    plt.show()
```

Visualize!
```{code-cell}
plot_dataset(dataset)
```

### Creating DataLoader

Since the MNIST dataset is an iteratable one, we can create pytorch DataLoader! 
```{code-cell}
import torch
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=20,
                                     shuffle=True,
                                     num_workers=1,
                                     pin_memory=True)
```

**Review**: the first argument is you dataset, and it can be anything but requires two attributes: [`__len__`](https://docs.python.org/3/reference/datamodel.html#object.__len__) and [`__getitem__`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__). In case you wonder, these attributes allow you to call `len(dataset)` and access dataset elements  by `dataset[X]` where `X` is an index integer.

#### Details (ignore if wished): other constructor arguments
The other constructor arguments used above are:
* `batch_size` ... the same of the subset data to be provided at once
* `shuffle` ... whether or not to randomize the choice of subset dataset (False will provide dataset
* `num_workers` ... number of parallel data-reader processes to be run (for making data read faster using `multiprocessing` module)
* `pin_memory` ... speed up data transfer to GPU by avoiding a necessity to copy data from pageable memory to page-locked (pinned) memory. Read [here](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/) for more details. If you are not sure about the details, set to `True` when using GPU. 

### Data streaming with `DataLoader`
So let's play with it! First of all, it has the concept of "length".
```{code-cell}
print('length of DataLoader:',len(loader))
print('By the way, batch size * length =', 20 * len(loader))
```

We know the data total statistics is 60,000 which coincides with the length of `DataLoader` instance and the batch size where the latter is the unit of batch data. **Yep, as you guessed**, `DataLoader` is iterable: 
```{code-cell}
# Create an iterator for playin in this notebook
from itertools import cycle
iter = cycle(loader)

for i in range(10):
    batch = next(iter)    
    print('Iteration',i)
    print(batch[1]) # accessing the labels
```

... and this is how `data` looks like:
```{code-cell}
print('Shape of an image batch data',batch[0].shape)
```

... which is quite naturally 20 of 28x28 image


## CIFAR10 

`CIFAR10` is yet another public dataset of 32x32 pixels RGB photographs. It contains 10 classes like MNIST but it is much more complicated than a gray scale, hand-written digits.
```{code-cell}
from torchvision import datasets, transforms
# Data file download directory
LOCAL_DATA_DIR = './cifar10-data'
# Create the dataset
dataset = datasets.CIFAR10(LOCAL_DATA_DIR, train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

plot_dataset(dataset)
```

Nothing new in terms of how-to, but let's also create a `DataLoader` with `CIFAR10`.
```{code-cell}
loader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,num_workers=1,pin_memory=True)

batch = next(cycle(loader))
```

Let's take a look at the `batch` data. Recall the shape of this image $(C,H,W)$ where `matplotlib.pyplot.imshow` takes the format $(H,W,C)$ just like how an ordinary photograph is presented. We use `torch.permute` function to swap the axis.
```{code-cell}
photos,labels=batch
for idx in range(len(photos)):
    photo = photos[idx].permute(1,2,0)
    label = labels[idx]
    print(dataset.classes[label])
    plt.imshow(photo)
    plt.show()
```




