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

# Up and Running with Python3

In this notebook we cover `numpy`, very widely used scientific python applications, it is so popular that most of other scientific applications are either built on top of `numpy` or at least support its data types. We cover the basics:
* [Introduction](#intro)
* [Data types](#datatype)
* [Array indexing, reshaping, slicing, masking](#array)
* [Saving array data](#save)
* [Linear algebra](#la)
* [Random numbers](#random)
* [Conclusion](#conclusion)

This notebook is based on [tutorial](https://github.com/marcodeltutto/Python-Tutorial-SBN-Workshop) by Marco Del Tutto for an introductory workshop.

<a href="intro"></a>
## NumPy: Getting Started

NumPy is a fundamental package for scientific computing with Python. 
Is a Python library adding support for multi-dimensional arrays and matrices, as well as many useful mathematical functions to operate on these arrays.

<img src="logos/775px-NumPy_logo.svg.png" style="width: 200px;">

```{code-cell}
import numpy as np
print(np.__version__)
```

In the previous notebook, we saw how to construct lists. Now, we will start from lists, and see how we can construct numpy arrays from them.
```{code-cell}
masses_list = [0.511, 105.66, 1.78e3]
masses_array = np.array(masses_list)
print(masses_array)
```

Multiply every element by a number:
```{code-cell}
masses_array_gev = masses_array * 1e-3
print(masses_array_gev)
```

To get the size of the array, you can use the `len()` function, or the `.size` attribute.

You can get the shape of the object by using the `.shape` attribute
```{code-cell}
# EXCERCISE: Check that len and size give the same result. What does shape return?
print('len is ', len(masses_array))
print('size is ', masses_array.size)
print('shape is ', masses_array.shape)
```

```{code-cell}
# EXCERCISE: Try np.linspace, np.zeros, np.ones
np.linspace(5, 15, 9)
np.zeros(9)
np.ones(9)
np.zeros((5, 4))
```

<a href="datatype"></a>
## NumPy DataTypes

Up until this point, we have been using the default datatypes that NumPy selects for arrays. In the cases for arange and linspace, the default types are integers. 

In the case of zeros and ones, the default type is floating point. Each of these functions has a `dtype` parameter. For example, we can look here and we see linspace has a `dtype` parameter and its default value is set to `None`. You can use this parameter to determine the datatype for each element in an array. Remember that each element must have the same datatype. 

At this [link](https://docs.scipy.org/doc/numpy/user/basics.types.html) you can find all the NumPy datatypes.

In the previous examples, we saw that the `ones` function and the `zeros` function return arrays that contain floating point values. 

You can change this and select the datatype that you want by setting a value for the `dtype` parameter. 
For example you can do `np.ones(9, dtype='int64')`.
```{code-cell}
# EXCERCISE: Create an array with zeros that has 11 elements, each of which is a 64-bit integer
a = np.zeros(11, dtype='int64')
print(a, type(a))
print(a.dtype)
```

And...there is also the _complex_ data type!

You can specify a complex type in python using `j` as imaginary number, as in `1+2j`.
```{code-cell}
# EXCERCISE: Try to add an imaginary number to a numpy array and print the array
masses_list = [0.511, 105.66, 1.78e3]
masses_list.append(1+2j)
masses_array = np.array(masses_list)
print(masses_array)
```

<a href="array"></a>
## Array Indexing, Reshaping, Slicing, Masking
```{code-cell}
masses_array = np.array([2.2, 4.7, 0.511, 1.28, 96, 105.66, 173e3, 4.18e3, 1.78e3, 0, 0, 91.19e3, 80.39e3, 124.97e3])
```

You can use negatixe index to start counting from the end of the array. 

For example, to select the last element:
```{code-cell}
print(masses_array[-1])
```
Or to select the penultimate element:
```{code-cell}
print(masses_array[-2])
```

### Reshape

The above array `masses_array` is a 1-D array with 14 elements in it. 
Numpy allows to resphape it easily. For example, we can transform it into a 2-D array with 7 columns and 2 rows.
```{code-cell}
masses_array_2d = np.reshape(masses_array, (7, 2))
print(masses_array_2d)
```

`reshape` also exists as an array attribute
```{code-cell}
masses_array_2d = masses_array.reshape((7,2))
print(masses_array_2d)
```

Exercise: try to reshape into `(7,3)`.
```{code-cell}
#Exercise: try to reshape into (7,3)
```

`reshape` function allows you to specify one of shape dimension value to be `-1`, which would mean "go figure out what it should be."
```{code-cell}
masses_array_2d = masses_array_2d.reshape((-1,7))
print(masses_array_2d.shape)
```

### Slicing

A basic slice syntax is `i:j:k` where `i` is the starting index, `j` is the stopping index, and `k` is the step:
```{code-cell}
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[1:7:2])
```

Now, if `i` is not given, it defaults to 0.

If `j` is not given, it defaults to the lenght of the array (call it `n`).

If `k` is not given it defaults to 1.


**Example** `i = 3`, `j` and `k` defaulted to `n` and 1:
```{code-cell}
print(x[3:])
```

**Example** `i` defaulted to 0, `j = 4` and `k` defaulted and 1:
```{code-cell}
print(x[:4])
```

**Example** `i` defaulted to 0, `j = 4` and `k = 2`:
```{code-cell}
print(x[:4:2])
```

### Masking
Let's start with a numpy array
```{code-cell}
vector = np.array([26, 14, 1, -28, 8, 7])
```

Then, we create a "mask". We construct a list with contains True and False values, depending if the elements of `vector` are divisbile or not by 7.
```{code-cell}
mask = 0 == (vector % 7)
print(mask)
```

Finally, we can applt this mask to our vector in order to select only elements that are divisible by 7:
```{code-cell}
print(vector[mask])
```

<a href="save"></a>
## Saving & Reading an array

It's useful to be able to save your array = data! So here's how you can save multiple arrays in a file.
```{code-cell}
np.savez('erase.npz', kazu=masses_array, daniel=masses_array_2d)
```

The above command saves `masses_array` and `masses_array_2d` data with keywords "kazu" and "daniel".

Confirm the file `erase.npz` is created. The choice of this filename is to remind our future-self that this can be removed :)
```{code-cell}
! ls -lht erase.npz 
```

Now let's re-read data from the file.
```{code-cell}
data = np.load('erase.npz')
type(data)
```

You can access `erase.npz` file contents by the keywords you defined upon saving the data.
```{code-cell}
print('shape of daniel',data['daniel'].shape)
print('contents of kazu')
print(data['kazu'])
```

<a href="la"></a>
## Linear Algebra

The `np.matrix` function returned a matrix from an array like object, or from a string of data. 

A matrix is a specialized 2D array that retains its 2D nature through operations. 

It has special operators such as asterisk for matrix multiplication, and a double asterisk for matrix power or matrix exponentiation operations.

Let's contruct the a CKM matrix:
```{code-cell}
ckm_matrix = np.matrix([[0.97427, 0.22534, 0.00351 ],
                        [0.22520, 0.97344, 0.0412  ],
                        [0.00867, 0.0404,  0.999146]])
print(ckm_matarix)
print(type(ckm_matrix))
```

Again, we can use the `.shape` attribute to see what is the shape of this matrix:
```{code-cell}
print(ckm_matrix.shape)
```
And also `ndim` to see the number of dimensions:
```{code-cell}
print(ckm_matrix.ndim)
```
Let's use the `help` function to see what opetations are available:
```{code-cell}
#help(np.matrix)
```

The transpose attribute .T to calculate the transpose of this matrix.

Next we'll use another attribute, .I, to calculate the inverse of this matrix. Notice that the inverse is calculated on my first matrix, and not upon the transform of my first matrix.

For example, is the transpose of the CKM matrix:
```{code-cell}
ckm_matrix.T
```

Let's check that the CKM matrix is unitary:
```{code-cell}
result = ckm_matrix * ckm_matrix.I.T
print(result)
```

<a href="random"></a>
## Random numbers
A random number generator is handy for simulations etc.
Here's an example of a **flat random number** between 0 to 1.
```{code-cell}
flat_random = np.random.random(100000)
print('shape',flat_random.shape)
print('mean',flat_random.mean(),'std',flat_random.std())
print('min',flat_random.min(),'max',flat_random.max())
```

... and there are others, like a normal distribution
```{code-cell}
flat_random = np.random.randn(100)
print('shape',flat_random.shape)
print('mean',flat_random.mean(),'std',flat_random.std())
print('min',flat_random.min(),'max',flat_random.max())
```

### Random seed
A reproducible behavior is important for many things including debugging of your code. For a random number generator, this is controlled by what's called _seed_. If you set the random number seed, then the sampled values from a distribution is predictable even though they may appear random. Let's give a shot!
```{code-cell}
SEED=123
np.random.seed(SEED)
```

Now let's sample 3 random values sampled from a normal distribution.
```{code-cell}
print(np.random.randn(3))
```

... and try again.
```{code-cell}
print(np.random.randn(3))
```

OK, so I don't know what values come out if we try yet another time. However, if we re-set he seed, we can expect the exact same values to be drawn.
```{code-cell}
SEED=123
np.random.seed(SEED)
print(np.random.randn(3))
```

_Voila_!

<a href="conclusion"></a>

# Conclusions
You know, you can also fly with Python. Let's try:
```{code-cell}
import antigravity
```


