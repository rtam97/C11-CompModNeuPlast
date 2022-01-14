# C11 - CompModNeuPlast

This repository contains the code and the results of the simulations ran as part of the course:

<p>
  <i>
  C11 - Computational Modeling of Neuronal Plasticity <br>
  Interdisciplinary Neuroscience Master<br>
  Goethe University Frankfurt<br>
  </i>
</p>

Professor: [Dr. Jochen Triesch](https://www.fias.science/en/neuroscience/research-groups/jochen-triesch/)

# Content

## :file_folder: Scripts

### `neurons.py`
This script contains a variety of functions, including equations for the simulated neurons and ways to generate stimuli 

### `solvers.py`
This script contains the functions used to numerically solve the neuron equations using the forward Euler method. At the moment there is a separate integration function for each neuron. In the future I would like to make a single function which can solve any neuron in the course... WIP!

### :file_folder: Excercises
In this directory lie all the individual scripts for each exercis in the course. In theory, one should be able to run them and obtain the figures found in *Results*

## :file_folder: Parameters
In this directory lie all the parameter files which are used to run the various python scripts. Because some excercises do not require much parameters, some lack a parameter file.

## :file_folder: Results
In this directory lie all the figures that have been created with the scripts found in *Exercises*. Maybe in the end I will upload the report here

---
