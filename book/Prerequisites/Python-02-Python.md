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

# Introduction to Python

Python is a modern, object-oriented scripting language. 
Developed in the late 1980s by Dutch research programmer Guido van Rossum. Guido wrote the first Python interpreter over his Christmas holiday in 1989. Guido remains Python's principal author to this day. The Python community refers to him as BDFL, Benevolent Dictator For Life. It is named after "Monty Python's Flying Circus" because of its developers' intent that programming should be fun.

**Python's core philosophy**: 
- Beautiful is better than ugly. 
- It's always a good idea to make your code elegant and readable. 
- Explicit is better than implicit. Don't make your readers guess what your code does, make it obvious. 
- Simple is better than complex. If you can make it simple, do. 
- Complex is better than complicated. If complexity is necessary, don't complicate it. 
- Readability counts. 

As a casual reader with a good understanding of the Python language, you should be able to understand the code with a minimal amount of effort. For complete philosophy, simply execute a command `import this`.  Python 3 is very close to Guido's ideal vision for Python. 

**Python is a rich language**, and it would be impossible to cover all of it in just one hour. But the online documentation is both accessible and exhaustive. 

**Python is a very powerful and versatile language.** It's supported on virtually all common operating systems and it's easy to learn and to support. 

<img src="logos/python-powered-h-140x182.png">

This notebook is partially taken from a [tutorial](https://github.com/marcodeltutto/Python-Tutorial-SBN-Workshop) by Marco Del Tutto for an introductory workshop.

## Hello World
```{code-cell}
print('Hello world!')
```

## Import a module from the Python library
```{code-cell}
import platform
print('We are using python', platform.python_version())
```

## Wait, what is "import"?
`import` is used to load a python _module_, and is probably the most important syntax in Python. A _module_ is a predefined set of tools (i.e. constants, functions, classes). In other languages, it's called sometimes a _package_ or a _library_. If you are familiar with C++ compiler, you could consider `import` close to `#include` preprocessor directive.

The syntax to load a certain module named `X` goes by `import X`. So in the above cell, we imported a module named `platform` which includes information about the very Python kernel we are running this program on.

Let's try another one.
```{code-cell}
import math
math.pi
```

**You can define your own module, and it's a good thing.** As you become more mature with Python (literally by the end of this notebook), you start defining many useful functions and may want to organize them under your own module name like `DanielRatnerLoveThis`. That's natural! We come back to this later in this notebook. 

## Datatypes

There are just a few fundamental data types in Python.

Python is designed to be extensible so it's easy to create your own types within its object system.
```{code-cell}
x = 7
print(type(x))

x = 7.0
print(type(x))

x = 1+2j
print(type(x))

x = '7'
print(type(x))

x = True
print(type(x))

x = None
print(type(x))
```

There is no `double` in Python. A `float` in Python has double precision (like a `double` in C). 

### Strings

Strings are objects, even literal strings. Here we start with literal string, the `'neutrino'` in quotes.
Since this is an object, we can run methods on it. Let's start with `capitalize()` here. 
```{code-cell}
x = 'neutrino'
print(x)

print(x.capitalize())

print(x.upper())

print(x.upper().lower())
```

You can treat strings as _array-like objects_ and access them using indexes with `[2]` to access the third index for example: 
```{code-cell}
# EXCERCISE: Access the fifth index of the above string
x[4]
```

You can print a string with the `print` function, and format it to also show values:
```{code-cell}
n_neutrinos = 3
x = 'We currently know there are {} neutrinos.'.format(n_neutrinos)
print(x)

# Or even easier with f-strings (Python >= 3.6)
x = f'Or maybe there are more than {n_neutrinos}?'
print(x)
```

### Lists
Lists in Python represent ordered sequences of values. Here is an example:
```{code-cell}
particles = ['up', 'down', 'electron', 'electron_neutrino',
             'charm', 'strange', 'muon', 'muon_neutrino', 
             'top', 'bottom', 'tau', 'tau_neutrino',
             'gluon', 'photon', 'z', 'w',
             'higgs']

masses = [2.2, 4.7, 0.511, 'less than 1 eV',
          1.28, 96, 105.66, 'less than 0.17 eV',
          173e3, 4.18e3, 1.78e3, 'less than 18.2 eV',
          0, 0, 91.19e3, 80.39e3,
          124.97e3]
```

Lists: `[]` - mutable

Tuples: `()` - unmutable

Here are some useful operations that can be applied on lists.
```{code-cell}
# Indexing
print(particles[5])
```

You can use negative indexes too. Negative indexes start counting from the end.
```{code-cell}
# EXERCISE: Try to slice the particle array with negative indexes.
print(particles[-1])
print(particles[-2])
```

You can also "slice" lists. More details on this will be given in the next notebook on numpy
```{code-cell}
# From the first element to the end
print(particles[1:])

# From the second to the fourth element 
print(particles[2:4])
```

With `len` you can get the lenght of the vector:
```{code-cell}
# Lenght
print(len(particles))
```

### Dictionaries
Dictionaries are a built-in Python data structure for mapping keys to values.
```{code-cell}
par_to_mass = {'electron_neutrino': 0, 
               'muon_neutrino': 0,
               'tau_neutrino': 0}
par_to_mass
```

```{code-cell}
# Query a dictionary
print('Electron-neutrino mass is', par_to_mass['electron_neutrino'])

# Change a value
par_to_mass['electron_neutrino'] = 'less than 1 eV'
print('No, we only know it is', par_to_mass['electron_neutrino'])
```

`get()` is a useful function that can set a default return value if a specified key does not exist.
```{code-cell}
par_to_mass.get('kazu','very heavy')
```

```{code-cell}
# Let's construct the full dictionary to go from particles to masses
par_to_mass = {}
for p, m in zip(particles, masses):
    par_to_mass[p] = m
```

To retrieve a list of keys and/or values from a dictionary:
- `.items()` applied to a dictionary return a pair of keys and values
- `.keys()` applied to a dictionary return all the keys
- `.values()` applied to a dictionary return all the values

Let's try looping over all key-value pairs using `items()`.
```{code-cell}
for key, value in par_to_mass.items():
    print(key, value)
# for key in par_to_mass.keys():
#     print(key)
# for value in par_to_mass.values():
#     print(value)
```

The `in` operator allows to check is a key is in the dictionary.
```{code-cell}
# EXCERCISE: Try to use "in" to check is a key is in the dictionary
print('Do I know the mass of Marco?', 'Marco' in par_to_mass)
```

### Lists and Dictionaries: Conclusions
Python provides a number of collection types useful for creating structured data. 

- The *list* type is a basic sequence. It's created using a pair of square brackets around a list of **objects** separated by commas. The list is mutable, which means that you may add, delete, and change values.

- A *dictionary* is a sequence of key-value pairs. In other languages, this may be called an associative array or a hashed array. It is indicated with curly brackets. 

- A *tuple* is like a list, but it's immutable. You cannot change it once it's been created. A tuple is created using parentheses. 

- A *set* is is an unordered list of unique values. It's useful for finding and operating upon unique values within a sequence. 

Any of these collection types may contain any object or type. 

## Blocks and Functions

Functions are the basic unit of reusable code in Python. So, let's take a look at how they work.
A couple of examples of functions in python: `round` and `abs`:
```{code-cell}
x = round(3.2)
y = abs(-3.6)
print('x =', x)
print('y =', y)
```

The `help()` function is possibly the most important Python function you can learn.
```{code-cell}
help(round)
```

```{code-cell}
# EXCERCISE:
# You can also get help in jupyter by using `?`. Try `round?`

```

Python differs from many languages in how **blocks** are delimited. Python does not use brackets or any special characters for blocks, instead it uses indentation.

A function is defined with the `def` keyword, then the name of the function, and then always parenthesis, even if it does not take any arguments. And any arguments are separated by commas inside of those parentheses. 
```{code-cell}
def check_temperature(T):
    '''
    Checks if the argon is in liquid state
    give a certain temperature T.
    '''
    T_c = 87 # K

    if T > T_c:
        print('Argon is NOT in liquid state!')
    else:
        print('Argon is in liquid state!')
        
check_temperature(88)
check_temperature(86)
```

It's a good idea to leave some comment in your code. Even better for it to show up on `help`!
```{code-cell}
help(check_temperature)
```

### Argument List
Let's define a simple function that prints out the mass of a particle.
```{code-cell}
def get_mass(particle_name):
    try:
        return par_to_mass[particle_name]
    except:
        print('Not found!')
```

Try!
```{code-cell}
get_mass(particle_name='higgs')
```

What if I would like to call this function for `higgs` and `electron`? I can certainly do:
```{code-cell}
get_mass('higgs')
get_mass('electron')
```

But that's a lot of typing on the user's end. Actually, Python functions allow variable length argument lists.

If you add an argument with an asterisk `*` before it (as done below with `*args`), this is a variable length argument list. It is a `tuple`.

Below we check if it has non-zero length, and then we loop over the arguments.

Note that we don't have to specify how many arguments or what kind of arguments we are passing. This feature is useful for situations where you want a function that may be called with different numbers of arguments.

```{code-cell}
def get_mass_2(*args):
    if len(args):
        for p in args:
            try: 
                print(p, 'mass is', par_to_mass[p],'MeV/c^2')
            except: 
                print(p, 'not found!')
```

Now a user can try:
```{code-cell}
get_mass_2('higgs','electron')
```

Or, one can also do:
```{code-cell}
names=['higgs','electron']
get_mass_2(names)
```

### Keyword Arguments

Keyword arguments are like list arguments that are dictionaries instead of tuples. This allows functions to have a variable number of named arguments. 

```{code-cell}
def get_mass_3(**kwargs):
    for k in kwargs:
        print('Particle', k, 'has mass', kwargs[k])
```

Let's try!
```{code-cell}
get_mass_3(Higgs=124970.0, Electron=0.511)

```

### Everything is an object
A list can contain a mix of different types of variables. It can also contain a function!

```{code-cell}
# Here it contains a string, a float, a dictionary, and a function!
miscellaneous = ['neutrino', 2.2, par_to_mass, get_mass]
```

Let's try!
```{code-cell}
miscellaneous[-1]('bottom')
```

## Classes

A class is the basis of all data in Python. 

Everything is an object in Python, and a class is how an object is defined.

A class definition beginning with the keyword `class`, the name of the class, and a colon, and then everything that defines the class is indented under that declaration. 

Inside the clasee you can have data in the form of **variables**, and **methods** in the form of function definitions.
```{code-cell}
class Computer:
    
    _name = 'dell xps15'
    _desktop = False
    
    def name(self):
        print('I am ', self._name, '!', sep='')
    
    def is_desktop(self):
        return self._desktop
```

You'll notice that:
- The first parameter of a method is always `self`. `self` is not a keyword. You can actually name that first parameter whatever you want to, but self is traditional and it is highly recommend that you use `self` so that as people are reading your code, they know what you're talking about. `self` is a reference to the object, not the class. And so, when an object is created from the class, `self` will reference that object. And then everything that references anything defined in the class is dereferenced off itself to get the instantiated object version of it. And the period operator is used to dereference the object. And the same is true outside of the class.
- All class variables start with an underscore `_`. This is a convention to remember ourselves that those variables belong to the class. Python doesn't have private variables, and so there is no way to actually prevent somebody from using these from outside of the class. But this indicates that it is a private variable and it should not be set or retrieved outside of the a setter or getter.

#### Next, we will create an instance of a class.  

An instance of a class is called an object. It's creating by calling the class itself as if it were a function. 
```{code-cell}
# EXCERCISE: Try to instantiate the class and call both class methods

me = Computer()
me.name()
print('This is a desktop.' if me.is_desktop() else 'This is a laptop.')
```

#### Class constructor  

The class constructor allows us to **initialize** class member variables when we instantiate the class.

The constructor can be defined in a special  class method name called `init`, with double underscores before and after: `__init__(self, var1, var2, ...)`.

```{code-cell}
# EXCERCISE: Add a constructor to the Computer class where you initialize the two variables _name and _near_det

class Computer:
    
    def __init__(self, name='dell xps15', desktop=False):
        self._name = name
        self._desktop = desktop
    
    def name(self):
        print('I am ', self._name, '!', sep='')
    
    def is_desktop(self):
        return self._desktop
    
me = Computer('imac', True)
me.name()
```

#### Let's print a class!

The method `__str__` allows to tell the class how it should behave inside a `print()` function.
```{code-cell}
# EXCERCISE: Try to add a the __str__ method and print some informaion from the Computer class, like this
# def __str__(self):
#     s = 'I am a desktop' if self._near_det else 'I am a laptop'
#     return f'Hi! I am {self._name}, and ' + s
# Also add a new method that returns the number of CPUs 

class Computer:
    
    def __init__(self, name='dell xps15', desktop=False):
        self._name = name
        self._desktop = desktop
        self._n_cpus = 8
    
    def name(self):
        print('I am ', self._name, '!', sep='')
    
    def is_desktop(self):
        return self._desktop
    
    def n_cpus(self):
        return self._n_cpus
    
    def __str__(self):
        s = 'I am a desktop.' if self._desktop else 'I am a laptop.'
        return f'Hi! I am {self._name}, and ' + s + f' I have {self.n_cpus()} CPUs.'

me = Computer('imac', True)
print(me)
```

Here we looked at a couple of special methods: `init` and `str`.

You can find a list of all the special method names in the [Python3 documentation](https://docs.python.org/3/reference/datamodel.html#basic-customization). 

Methods are the primary interface for classes and objects. They work exactly like functions, except they are bound to the object through their first argument, commonly named self.

### Class Inheritance

Class inheritance is a fundamental part of object-oriented programming.

This allows you to extend your classes by driving properties and methods from parent classes.
```{code-cell}
class iMac(Computer):
    def __init__(self):
        self._name = 'imac'
        self._desktop = True 
        self._n_cpus = 16

s = iMac()
print(s)

 
class XPS15(Computer):
    def __init__(self):
        super().__init__()

i = XPS15()
print(i)
```

The `super()` function above allows to call the parent class (`Computer`). `super().__init__()` calls the constructor of the `Computer` class.
```{code-cell}
# EXCERCISE: (my) xps15 has also 1 GPU. Extend the XPS15 class to also return the number of GPUs

class XPS15(Computer):
    def __init__(self):
        super().__init__('dell xps15',False)
        self._n_gpus = 1
    
    def n_gpus(self):
        return self._n_gpus
    
    def __str__(self):
        s = 'I am a desktop.' if self._desktop else 'I am a laptop.'
        return f'Hi! I am {self._name}, and ' + s + f' I have {self.n_cpus()} CPUs and {self._n_gpus} GPUs.'

s = iMac()
print(s)

i = XPS15()
print(i)
```

### Application: a variable collections
In C++, for quick encapsulation, we often use a _struct_ container. A hacky but handy way of achieving something similar in Python is to define an empty class, and dyncamically attach attributes.
```{code-cell}
class BLOB:
    pass

blob=BLOB()
blob.data   = XPS15()
blob.number = 2
```

#### `hasattr` and `getattr`
Since this blob object is dynamic, you might want a capability to check, when given a blob object, a certain attribute exists or not. Python has a handy built-in functions `hasattr` and `getattr` ("attr" stands for "attributes").
```{code-cell}
print('does it have "tracy"?', hasattr(blob, "tracy") )
print('\n__str__ of tracy...', getattr(blob,"tracy", 'no it does not exist'))
print('\ndoes it have "data"?', hasattr(blob, "data" ) )
print('\n__str__ of data...', getattr(blob, "data", 'no it does not exist'))
```

### Application: an iteratable class

An array-like data container in which individual data element can be accessed by an index (i.e. "[]") or iterate over would be very handy. Such a structure also helps to generalize a design of data streaming tools. We practice to design such a Python class below. 

First, let's define a iterable container. For an iteration in Python, you need two built-in functions: a _len_ which returns the "length of array-like data" and _getitem_ which arrows a random-access operator (i.e. "[ ]"). 
```{code-cell}
class dataset:
    
    def __init__(self):
        self._data = tuple(range(100))
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self,index):
        return self._data[index]
```

The above example class `dataset` constructs a simple data, an array of length 100 filled with numbers 0 to 99. The length of the dataset is accordingly 100, and the random access operator returns the corresponding index entry of the array.
```{code-cell}
data = dataset()
print('Length:',len(data))
print('10th element:',data[10])
```

You can also slice this data object
```{code-cell}
data[10:20]
```

Here's how you can create an iterator for an iterable object using _iter_ built-in function.
```{code-cell}
iter(data)
```

You can move the iterator to the next object using the _next_ built-in function. 
```{code-cell}
it = iter(data)
print(next(it), next(it), next(it))
```

## List Comprehensions 

List comprehensions are one of Python's frequently used features. The easiest way to understand them is probably to just look at a an example. 

Let's assume we want to calculate all the squarer of the numbers between 0 and 9. With a `for` loop we would do:
```{code-cell}
squares = []
for n in range(10):
    squares.append(n**2)
print(squares)
```

Instead, with list comprehension we can just do:
```{code-cell}
squares = [n**2 for n in range(10)]
print(squares)
```

Let's compare the timing:
```{code-cell}
squares = []
%timeit for n in range(1000): squares.append(n**2)
%timeit squares = [n**2 for n in range(1000)]
```

It can be also used with an if statement. Suppose we have a list of particle names:
```{code-cell}
particles = ['up', 'down', 'electron', 'electron_neutrino',
             'charm', 'strange', 'muon', 'muon_neutrino', 
             'top', 'bottom', 'tau', 'tau_neutrino',
             'gluon', 'photon', 'z', 'w',
             'higgs']
```

Here is how we can create a sub-list of names with less than 6 characters using a list comprehension.
```{code-cell}
short_particles = [particle for particle in particles if len(particle) < 6]
print(short_particles)
```

Another exercise to create a sublist of particles that has 'o' in it:
```{code-cell}
# EXCERCISE:
# Get all the particles that contain character 'o', and capitalize them.

selected = [particle.capitalize() for particle in particles if 'o' in particle]
print(selected)
```

## Dict Comprehensions

Dict comprehensions expands the idea of a list comprehension but for python dictionaries!

Lets create a dictionary of the squares of numbers 0 through 9 inclusive, could write a loop like this:
```{code-cell}
squares = {}
for n in range(10):
    squares[n] = n**2
print(squares)
```

Or we can write it as a dict comprehension as follows (note the special syntax "key: value" to assign the key and value of the dict)
```{code-cell}
squares = { n: n**2 for n in range(10) }
print(squares)
```

Like it was the case for a `list`, we can combine with an if statement.
```{code-cell}
squares = { n: n**2 for n in range(10) if n%2 == 0}
print(squares)
```

## Defining your module
Now that we know all basics, let's come back to defining our own module that can be loaded through `import`!

There are two ways to do this: a module can be deifned within a single script OR within a directory with multiple scripts. Let's practice both cases.

### A script as a module
You can import a script with `.py` suffix as a module. In this same directory, we prepared `DanielRatner.py`. It has a minimal content (unlike real Daniel who is full of features).

```{code-cell}
print(open('DanielRatner.py','r').read())
```

We can import `DanielRatner` as a module, and access attributes under this.
```{code-cell}
import DanielRatner
print('age:',DanielRatner.age)
```

We can also import a specific attribute from the module.
```{code-cell}
from DanielRatner import trend
print('trend:',trend)
```

### A directory as a module
As you go down the path, you might end up defining 100 useful functions, and maybe group them into multiple python scripts, yet you may want to group them into one module. You can then define a whole directory containing multiple Python script as a module. 

We prepare another example to exercise importing a whole directory as a module.
The directory name is called `RealDanielRatner` and contains three python scripts.

- `__init__.py` ... signifies to Python that this directory should be treated as a module. Define what attributes to be available within the module.
- `age.py` ... first module that contains Daniel's real age
- `favorite.py` ... second module that contains Daniel's favorite
- `trend.py` ... third module that contains Daniel's real trend
```{code-cell}
! ls RealDanielRatner
```

The contents of `__init__.py` defines what should be available under `RealDanielRatner` module.
```{code-cell}
print(open('RealDanielRatner/__init__.py','r').read())
```

The rest of scripts contain just 1 line information about Daniel.
```{code-cell}
print(open('RealDanielRatner/age.py','r').read())
print(open('RealDanielRatner/favorite.py','r').read())
print(open('RealDanielRatner/trend.py','r').read())
```

Now let's import `RealDanielRatner`.
```{code-cell}
import RealDanielRatner
help(RealDanielRatner)
```

Note that `age` and `trend` are available in `DATA` field and can be accessed directly.
```{code-cell}
print('age:',RealDanielRatner.age)
print('trend:',RealDanielRatner.trend)
```

... but not `favorite` which is recognized as a module under `RealDanielRatner`.
```{code-cell}
hasattr(RealDanielRatner,'favorite')
```

This is because `__init__.py`, which defines `RealDanielRatner` module from the whole directory, did not include a syntax to `import favorite`. Here is the content of `__init__.py`.
```{code-cell}
! cat RealDanielRatner/__init__.py
```

But you can certainly `from RealDanielRatner import favorite`, which would mean to load `favorite` module from `RealDanielRatner` (=directory).
```{code-cell}
from RealDanielRatner import favorite
print('favorite:',favorite.favorite)
```

### Renaming a module
Finally, `RealDanielRatner` is ... too long, right? You can rename the module upon import:
```{code-cell}
import RealDanielRatner as dan
print('Daniels age is',dan.age,'and his trend:',dan.trend)
```

### Where are the module source?
So we just demonstrated that we could import `DanielRatner` from the `DanielRatner.py` in the same directory. But would the same work if `DanielRatner.py` is somewhere else? Where does `import` syntax looks up the location of the source modules?

The answer is your current directory and a list of paths in `sys.path`
```{code-cell}
import sys
sys.path
```

If you wonder where an imported module came from, you can look up a `__file__` attribute where it is available. For instance,
```{code-cell}
print(platform.__file__)
```

... which tells you where the `platform` module we imporeted first in this notebook is defined.

## Conclusions

And now, just for fun, let's import this.

```{code-cell}
import this 
```

