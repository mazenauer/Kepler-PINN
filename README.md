# Kepler-PINN

This repository accompanies a semester thesis on physics-informed neural networks to solve Kepler problems.
The file odeintegrator.py includes a script to solve the IVP with a symplectic Runge-Kutta method to generate a high-fidelity solution.
The file twoBodyPINN.py includes the PINN class and some help functions to train the network and show the progress. The data folder contains the data that is used in the sensitivity analysis.

The demo_notebook and sensitivity_demo are meant to reproduce the plots that are featured in the thesis. 

Abstract:
*In this semester thesis the Kepler problem and physics-informed neural networks (PINNs) are introduced.
The main result is a python class PINN which can solve the Kepler problem for given initial values and
time interval by neural network approximation. A symplectic Runge-Kutta solver delivers a high-fidelity
solution for evaluation of the accuracy. In general the network generally gives reliable results however
the optimal hyper-parameters depend on the initial values. The accuracy of the prediction is influenced
by higher order interactions of the hyper-parameters which make an initial educated guess difficult. On
the upside the PINN is flexible, fast, scaleable, and generalizeable to more than two bodies.*
