DT
===========================

This project simulates the R CMa system based on its main parameters,
containing masses, radii, orbital radii, effective temperatures and luminosities of the two stars. 
The main parameters were optimised based on the predicted light flux and RV of the simulated system. 

###########environment
python 3.10

###########Dependent library
math
random
numpy
matplort
scipy
json
time

###########Directory
├── Readme.md                   // help
├── Calculation.py              // tool
├── RCMa_system.py              
├── RV.txt                      // radial velocity data
├── SimulatedAnnealing.py
├── V.dat                       // photometric data
├── solutions.json (output)     // json file that will be generated in the program
└── main.py                     // main program
