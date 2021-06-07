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

# Jupyter: notebook of code

Probably many of you know know this software better than I know :) but let's share 5 minutes together to make sure we all are on the same page. 

Jupyter notebook (formerly known as IPython notebook) has been developed for the concept of _literate programming_ and it has become extremely popular within last seveal years ([article on Nature](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)). As its name says ("notebook"), it is designed for users to program with readability and modularization in mind. 

In a notebook, individual block of code execution is done within a _cell_. All cells in the same notebook live within the same process and namespace scope. You can put explanation of the code (or beyond, like physics for which the code is developed) in a _markdown_ cell. Such explanation can go per code cell. We don't write a software in a notebook, but usually a higher level program (such as _main_ function) with an explanation of what's going on.

Though originally developed for Julia, Python and R (hence "Jupyter"), now it supports all kinds of programming language including shell and even fortran(!). Many scientific blog posts are also done in a notebook and blogs to teach you about Jupyter (like [this one](http://arogozhnikov.github.io/2016/09/10/jupyter-features.html#Writing-formulae-in-latex)).

## Default code engine
When you start a notebook, you choose the default backend of the notebook. Pick Python3!

```{code-cell}
import numpy as np
```

Note that the state of the process are shared among cells. What I just imported is accessible in the next cell.

```{code-cell}
print(np)
np='foo' # after you execute this cell, np no longer points to numpy module
print(np)
```

... which means the execution order of cells matter (yes, you can execute whichever cell in any order you want). The object `np` no longer points to a numpy module.

## Shell commands
You can run shell commands with `!` in front.

```{code-cell}
!ls $HOME
```

So if you want to install another python module and feel lazy, you can just execute `!pip install --user whatever` within a cell. 

## Different language
You can switch to a different language within a cell by specifying with `%%`, given that the language is supported by your environment. The software container we use doesn't have much options, but we got bash ;)

```{code-cell}
%%bash
ls $HOME
```

## Modifyng shell environment
Another thing we often feel lazy about is to change the shell environment variable. You can do this within the notebook like shown below (or stop the notebook, change environment value, then restart notebook ... not good for lazy people).

```{code-cell}
%env CUDA_VISIBLE_DEVICES=0
```

Now try:
```{code-cell}
! echo $CUDA_VISIBLE_DEVICES
```

_Voila!_ (you should see `0`) ... but note, the following is not the same:

```{code-cell}
! export CUDA_VISIBLE_DEVICES=1
```

Let's check:
```{code-cell}
! echo $CUDA_VISIBLE_DEVICES
```

You should get `0`, and that's because `! export ` executes the command in a _sub-shell_ and the scope is only within the cell.

## Run a python script
`%run` is a handy command to run a python script (or even jupyter notebook actually!). Let's create a simply python script.

```{code-cell}
! echo "print('hello world!')" >> hello.py
```

and run it:
```{code-cell}
%run hello.py
```

## Time your program
You have a code execution cell and want to measure how much time it takes. Sure you can add such profiling feature to your code, but here's how you can do in the notebook using `%%time`.

```{code-cell}
%%time
import numpy as np
sum = np.sum(np.ones([1000,1000],dtype=np.float32))
```

## Latex
Jupyter supports `MathJax`, and also you can just run latex

```{code-cell}
%%latex

I have to write a formula like $\sin^22\theta\left(\frac{1.27\Delta m^2L}{E_\nu}\right)$ to get Ph.D.
```

## Executing a web script
Well, all of these fun things are running using `javascript` on `html`... can we run our own webscript execution command? Sure we can! Here's an example to hide all Jupyter code blocks using that feature.


```{code-cell}
from IPython.display import HTML
HTML('''<script>
code_show=false; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
```