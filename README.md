# Self-organized synergistic interdependencies
This repository contains the source code to reproduce the figures in the manuscript:
["Heterophily as a generative mechanism for self-organized synergistic interdependencies"](https://arxiv.org/html/2604.11545v1)

## Short guide
To reproduce the figures presented in the paper, check `notebooks/figures.jl`: this Pluto notebook contains all the figures.
The data used for the figures is stored in the folder `data/`.

Each Pluto notebook automatically instantiates the environment and loads all the functions in `src/` by including `HeterophilySynergy.jl`.

### Reproducibility
To regenerate the data or collect your own data, please use the notebooks in `data_collection/`.
To reproduce everything exactly, check the starting seed numbers embedded in the filenames in `data/` and use the data collection cells in the notebooks in `data_collection/`.
If you have suggestions for improving reproducibility, please get in touch!

## Why Pluto notebooks?
For each Pluto notebook (just a `.jl` file) there is an identical `.html` file in the same folder.
This is just so that if you don't use julia you can just open the `.html` files on your browser, check the functions or even run the Pluto notebook online via Binder.

## Just interested in the model?
The best way to have fun with the model is to go through a small tutorial I have made: `examples/example_simulation.jl` (this is a script, not a notebook).

## Any suggestions?
Please do not hesitate to contact me!

## Contact
**Email:** ec627@sussex.ac.uk
