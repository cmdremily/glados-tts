# GLaDOS Text-to-speech (TTS) Voice Generator

A "librarificated" version of https://github.com/R2D2FISH/glados-tts, see the README there for additional info.

Main differences are:
 - There's no remote server or stand-alone program.
 - `pip install .` will install a python module that can be imported without module import errors.
 - Unnecessary dependencies and unused code has been removed.
 - Various cleanups and optimizations.


To run the TTS on an example sentence to see if it works, just run:

```console
python3 example.py
```

## Installation Instruction

1. Install the required Python packages, e.g., by running `pip install -r requirements.txt`.
1. Download the model files from [`Google Drive`](https://drive.google.com/file/d/1TRJtctjETgVVD5p7frSVPmgw8z8FFtjD/view?usp=sharing) and unzip into the models folder.
