# Installation

This document describes the steps to install the tools we will use throughout
the course (VSCode, Python 3.8, PyTorch, Conda, and IPython) on Windows 10.
All these tools are also available on other operating systems (Linux, Mac).

<!---
* [Visual Studio Code (VSCode)][VSCode] is the suggested code editor for this
  course
* [Conda][conda] is a package manager for Python.
-->


## Miniconda

Donwload and run the **Python 3.8 Miniconda3 Windows 64-bit** installer from [this
website][miniconda-windows-installers].  You can stick with the default options
proposed during the installation process.

## VSCode

Download the **64 bit User Installer for Windows** from the [VSCode download
webpage][VSCode-downloads].  Run the installer, agree to the license, stick
with the default options.  Finally, open VSCode.

### VSCode Python Extension

Follow the steps described on the [VSCode Python Extension website](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

### Conda Environment

Then open VSCode and follow the steps:
<!---
* Open a new `test.py` file.  VSCode should detect `test.py` as a
  Python file and propose to install the **Python extention**.  If not, you can
  always do it manually (select the *extention* tab on the left and
  search for Python), or
-->
* Open the terminal using `` CTRL+` `` (or from the menu: `View->Terminal`).
  If you encounter the following error:

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



[VSCode]: https://code.visualstudio.com/
[VSCode-downloads]: https://code.visualstudio.com/Download
[miniconda-windows-installers]: https://docs.conda.io/en/latest/miniconda.html#windows-installers