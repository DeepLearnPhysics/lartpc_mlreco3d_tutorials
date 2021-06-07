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

# Pytorch: Machine Learning library

[Pytorch](https://pytorch.org/) is one of open-source, modern deep learning libraries out there and what we will use in this workshop. Other popular libraries include [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io), [MXNet](https://mxnet.apache.org), [Spark ML](https://spark.apache.org/mllib/), etc. ...

All of those libraries works very similar in terms of implementing your neural network architecture. If you are new, probably any of Pytorch/Keras/Tensorflow would work well with lots of guidance/examples/discussion-forums online! Common things you have to learn include:

1. [Data types](#datatype) (typically arbitrary dimension matrix, or _tensor_ )
2. [Data loading tools](#dataloader) (streamline prepping data into appropraite types from input files)
3. [Chaining operations](#graph) = a _computation graph_ 

In this notebook, we cover the basics part in each of topics above.

<a href="datatype"></a>
## 1. Tensor data types in PyTorch
In `pytorch`, we use `torch.Tensor` object to represent data matrix. It is a lot like `numpy` array but not quite the same. `torch` provide APIs to easily convert data between `numpy` array and `torch.Tensor`. Let's play a little bit.

```{code-cell}
from __future__ import print_function
import numpy as np
import torch
SEED=123
np.random.seed(SEED)
torch.manual_seed(SEED)
```

... yep, that's how we set pytorch random number seed! (see Python-03-Numpy if you don't know about a seed)

### Creating a torch.Tensor

Pytorch provides constructors similar to numpy (and named same way where possible to avoid users having to look-up function names). Here are some examples.
```{code-cell}
# Tensor of 0s = numpy.zeros
t=torch.zeros(2,3)
print('torch.zeros:\n',t)

# Tensor of 1s = numpy.ones
t=torch.ones(2,3)
print('\ntorch.ones:\n',t)

# Tensor from a sequential integers = numpy.arange
t=torch.arange(0,6,1).reshape(2,3).float()
print('\ntorch.arange:\n',t)

# Normal distribution centered at 0.0 and sigma=1.0 = numpy.rand.randn
t=torch.randn(2,3)
print('\ntorch.randn:\n',t)
```

... or you can create from a simple list, tuple, and numpy arrays.
```{code-cell}
# Create numpy array
data_np = np.zeros([10,10],dtype=np.float32)
# Fill something
np.fill_diagonal(data_np,1.)
print('Numpy data\n',data_np)

# Create torch.Tensor
data_torch = torch.Tensor(data_np)
print('\ntorch.Tensor data\n',data_torch)

# One can make also from a list
data_list = [1,2,3]
data_list_torch = torch.Tensor(data_list)
print('\nPython list :',data_list)
print('torch.Tensor:',data_list_torch)
```

Converting back from `torch.Tensor` to a numpy array can be easily done
```{code-cell}
# Bringing back into numpy array
data_np = data_torch.numpy()
print('\nNumpy data (converted back from torch.Tensor)\n',data_np)
```

Ordinary operations to an array also exists like `numpy`.
```{code-cell}
# mean & std
print('mean',data_torch.mean(),'std',data_torch.std(),'sum',data_torch.sum())
```

We see the return of those functions (`mean`,`std`,`sum`) are tensor objects. If you would like a single scalar value, you can call `item` function.
```{code-cell}
# mean & std
print('mean',data_torch.mean().item(),'std',data_torch.std().item(),'sum',data_torch.sum().item())
```

### Tensor addition and multiplication
Common operations include element-wise multiplication, matrix multiplication, and reshaping. Read the [documentation](https://pytorch.org/docs/stable/tensors.html) to find the right function for what you want to do!
```{code-cell}
# Two matrices 
data_a = np.zeros([3,3],dtype=np.float32)
data_b = np.zeros([3,3],dtype=np.float32)
np.fill_diagonal(data_a,1.)
data_b[0,:]=1.
# print them
print('Two numpy matrices')
print(data_a)
print(data_b,'\n')

# Make torch.Tensor
torch_a = torch.Tensor(data_a)
torch_b = torch.Tensor(data_b)

print('torch.Tensor element-wise multiplication:')
print(torch_a*torch_b)

print('\ntorch.Tensor matrix multiplication:')
print(torch_a.matmul(torch_b))

print('\ntorch.Tensor matrix addition:')
print(torch_a-torch_b)

print('\nadding a scalar 1:')
print(torch_a+1)
```

### Reshaping

You can access the tensor shape via `.shape` attribute like numpy
```{code-cell}
print('torch_a shape:',torch_a.shape)
print('The 0th dimension size:',torch_a.shape[0])
```

Similarly, there is a `reshape` function
```{code-cell}
torch_a.reshape(1,9).shape
```

... and you can also use -1 in the same way you used for numpy
```{code-cell}
torch_a.reshape(-1,3).shape
```

### Indexing (Slicing)

We can use a similar indexing trick like we tried with a numpy array
```{code-cell}
torch_a[0,:]
```
or a boolean mask generation
```{code-cell}
mask = torch_a == 0.
mask
```
... and slicing with it using `masked_select` function
```{code-cell}
torch_a.masked_select(~mask)
```

<a href="dataloader"></a>
## 2. Data loading tools in Pytorch

In Python-02-Python, we covered an iteratable class and how it could be useful to generalize a design of data access tools. Pytorch (and any other ML libraries out there) provides a generalized tool to interface such iteratable data instance called `DataLoader`. Desired capabilities of such tools include ability to choose random vs. ordered subset in data, parallelized workers to simultaneously prepare multiple batch data, etc..

Let's practice the use of `DataLoader`. 

First, we define the same iteretable class mentioned in Python-02-Python notebook.
```{code-cell}
class dataset:
    
    def __init__(self):
        self._data = tuple(range(100))
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self,index):
        return self._data[index]
    
data = dataset()
```

Here is how you can instantiate a `DataLoader`. We construct an instance called `loader` that can automatically packs 10 elements of data (`batch_size=10`) that is randomly selected (`shuffle=True`) using 1 parallel worker to prepare such data (`num_workers=1`).
```{code-cell}
from torch.utils.data import DataLoader
loader = DataLoader(data,batch_size=10,shuffle=True,num_workers=1)
```

The dataloader itself is an iterable object. We created a dataloader with batch size 10 where the dataset instance has the length 100. This means, if we iterate on the dataloader instance, we get 10 separate batch data. 
```{code-cell}
for index, batch_data in enumerate(loader):
    print('Batch entry',index,'... batch data',batch_data)
```

We can see that data elements are chosen randomly as we chose "shuffle=True". Does this cover all data elements in the dataset? Let's check this by combining all iterated data.
```{code-cell}
data_collection = []
for index,batch_data in enumerate(loader):
    data_collection += [int(v) for v in batch_data]
    
import numpy as np
np.unique(data_collection)
```

This covers the minimal concept of `DataLoader` you need to know in order to follow the workshop. You can read more about `DataLoader` in pytorch documentation [here](https://pytorch.org/docs/stable/data.html) and also more extended example in [their tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) if you are interested in exploring yourself.

<a href="graph"></a>
## 3. Computation graph

The last point to cover is how to chain modularized mathematical operations. 

To get started, let's introduce a few, well used mathematical operations in pytorch.

* `torch.nn.ReLU` ([link](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)) ... a function that takes an input tenor and outputs a tensor of the same shape where elements are 0 if the corresponding input element has a value below 0, and otherwise the same value.
* `torch.nn.Softmax` ([link](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax)) ... a function that applies a [softmax function](https://en.wikipedia.org/wiki/Softmax_function) on the specified dimension of an input data.
* `torch.nn.MaxPool2d` ([link](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)) ... a function that down-sample the input matrix by taking maximum value from sub-matrices of a specified shape.

Let's see what each of these functions do first using a simple 2D matrix data.

```{code-cell}
# Create a 2D tensor of shape (1,5,5) with some negative and positive values
data = torch.randn(25).reshape(1,5,5)
data
```

Here's how `ReLU` works
```{code-cell}
op0 = torch.nn.ReLU()
op0(data)
```

Here's how `Softmax` works
```{code-cell}
op1 = torch.nn.Softmax(dim=2)
op1(data)
```

Here's how `MaxPool2d` works with a kernel shape (5,1)
```{code-cell}
op2 = torch.nn.MaxPool2d(kernel_size=(1,5))
op2(data)
```

So if we want to define a computation graph that applies these operations in a sequential order, we could try:
```{code-cell}
op2(op1(op0(data)))
```

Pytorch provides tools called _containers_ to make this easy. Let's try `torch.nn.Sequential` (see different type of containers [here](https://pytorch.org/docs/stable/nn.html#containers)).
```{code-cell}
myop = torch.nn.Sequential(op0,op1,op2)
myop(data)
```

We might wonder "Can I add a custom operation to this graph?" Yes, we can add any _module_ that inherits from `torch.nn.Module` class. Let's define one for ourself.
```{code-cell}
class AddOne(torch.nn.Module):

    # always call the base class constructor for defining your torch.nn.Module inherit class!
    def __init__(self):
        super().__init__()
        
    # forward needs to be defined. This is called by "()" function call.
    def forward(self,input):
        
        return input + 1;
```

Now let's add our operation
```{code-cell}
myop = torch.nn.Sequential(op0,op1,op2,AddOne())
myop(data)
```

Of course, you can also embed `op0`, `op1`, and `op2` inside one module.
```{code-cell}
class MyOp(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self._sequence = torch.nn.Sequential(torch.nn.ReLU(), 
                                             torch.nn.Softmax(dim=2), 
                                             torch.nn.MaxPool2d(kernel_size=(1,5)),
                                             AddOne(),
                                            )
        
    def forward(self,input):
        
        return self._sequence(input)
```

Let's try using it.
```{code-cell}
myop = MyOp()
myop(data)
```

## Extra: GPU acceleration

**This section only works if you run this notebook on a GPU-enabled machine (not on the binder unfortunately)**

Putting `torch.Tensor` on GPU is as easy as calling `.cuda()` function (and if you want to bring it back to cpu, call `.cpu()` on a `cuda.Tensor`). Let's do a simple speed comparison. 

Create two arrays with an identical data type, shape, and values.
```{code-cell}
# Create 1000x1000 matrix
data_np=np.zeros([1000,1000],dtype=np.float32)
data_cpu = torch.Tensor(data_np).cpu()
#data_gpu = torch.Tensor(data_np).cuda()
```

Time fifth power of the matrix on CPU
```{code-cell}
%%timeit
mean = (data_cpu ** 5).mean().item()
```
... and next on GPU
```{code-cell}
%%timeit
mean = (data_gpu ** 5).mean().item()
```

... which is more than x10 faster than the cpu counter part :)

But there's a catch you should be aware! Preparing a data on GPU does take time because data needs to be sent to GPU, which could take some time. Let's compare the time it takes to create a tensor on CPU v.s. GPU.
```{code-cell}
%%timeit
data_np=np.zeros([1000,1000],dtype=np.float32)
data_cpu = torch.Tensor(data_np).cpu()
```

```{code-cell}
%%timeit
#data_np=np.zeros([1000,1000],dtype=np.float32)
#data_gpu = torch.Tensor(data_np).cuda()
```
As you can see, it takes nearly 10 times longer time to create this particular data tensor on our GPU. This speed depends on many factors including your hardware configuration (e.g. CPU-GPU communication via PCI-e or NVLINK). It makes sense to move computation that takes longer than this data transfer time to perform on GPU.