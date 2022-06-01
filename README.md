# Kepler-PINN

This repository accompanies a semester thesis on physics-informed neural networks to solve Kepler problems.
The file odeintegrator.py includes a script to solve the IVP with a symplectic Runge-Kutta method to generate a high-fidelity solution.
The file twoBodyPINN.py includes the PINN class and some help functions to train the network and show the progress.

The demo_notebook and sensitivity_demo are meant to reproduce the plots that are featured in the thesis. 

Abstract:
*In this semester thesis I introduce the Kepler problem and physics-informed neural networks (PINNs). The main result is a python class PINN that can solve the Kepler problem for given initial values over a given time interval by neural network approximation. For direct evaluation of the accuracy a symplectic Runge-Kutta solver delivers a high-fidelity solution. The network generally gives reliable results but the optimal hyper-parameters depend on the initial values. The accuracy of the prediction is influenced by higher order interactions of the hyper-parameters which makes it difficult to start with an educated guess. On the other hand, the PINN is flexible, fast, scaleable, and generalizeable to more than two bodies.*
