"""Physical constants used throughout the Monte Carlo simulation.

Note:
    All angles are in radians
    All distances are in cm
    All energies (momenta, masses) are in MeV
    All deltas are in percent
    All densities (thicknesses) are in g/cm3 (g/cm2)
    All B fields are in kG
"""

import math

# Particle masses (MeV)
Me = 0.51099906
Me2 = Me ** 2

Mp = 938.27231
Mp2 = Mp ** 2

Mn = 939.56563
Mn2 = Mn ** 2

Mpi = 139.56995
Mpi2 = Mpi ** 2

Mk = 493.677
Mk2 = Mk ** 2

Md = 1875.613
Md2 = Md ** 2

Mlambda = 1115.68
Msigma0 = 1192.64
Msigma_minus = 1197.45

amu = 931.49432
hbarc = 197.327053

pi = math.pi
twopi = 2.0 * pi
alpha = 1.0 / 137.0359895
alpi = alpha / pi
degrad = 180.0 / pi
euler = 0.577215665
