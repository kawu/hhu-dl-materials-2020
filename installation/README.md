# Installation

This document describes the steps to install the tools we will use throughout
the course (VSCode, Python 3.8, PyTorch, Miniconda, and IPython) on Windows 10.
All these tools are also available on other operating systems (Linux, Mac).

* [Visual Studio Code (VSCode)][VSCode] is the suggested code editor for this
  course
* [Miniconda][miniconda] is a minimal distribution of the conda package manager
  for Python.  TODO


## Miniconda

Donwload and run the [Python 3.8 Windows Installer][miniconda-3.8-installer].
You can stick with the default options proposed during the installation
process.

## VSCode

Download the [User 64 bit installer][VSCode-user-64-installer].  Run it, agree
to the license, stick with the default options.

Open VSCode and then:
* Open a new `test.py` file.  VSCode should detect `test.py` as a
  Python file and propose to install the Python extention.  If not, you can
  always do it manually (select the *extention* tab on the left and
  search for Python).
* Open the terminal using `CTRL+\`` (or from the menu: `View-\>Terminal`).
  If you encounter the following error:
```
```
  switch from PowerShell to Command Prompt, as described [on
  stackoverflow](https://stackoverflow.com/questions/54828713/working-with-anaconda-in-visual-studio-code).
  Then, opening the terminal should automatically activate the `base`
  conda environment.
* In the terminal, create a new environment which you will use throughout the
  practical sessions:

      conda create --name dlnlp python=3.8

  The `python=3.8` option should ensure that we you the latest Python 3.8
  version availabe.
* Finally, activate your new conda environment, `dlnlp`, in VSCode by selecting
  it in the lower-left cover.  To see `dlnlp` in the list, you may need to
  restart VSCode first.
  

# PyTorch

Use the following command to install PyTorch (without CUDA support):

    TODO

Then, you can start a `python` session and type:
```python
>>> import torch
```
to verify that it works.
