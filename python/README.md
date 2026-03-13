# mc-single-arm — Python rewrite

This directory contains a complete Python (3.8+) rewrite of the Fortran
`mc_single_arm` Monte Carlo programme.  All physics and geometry logic is
faithfully translated from the original Fortran source in `../src/`.

## Requirements

```
numpy>=1.20
```

Install dependencies with:

```bash
pip install numpy
```

No compiled extensions are required.

## Running

```bash
cd python
python mc_single_arm.py <input_stem>
```

The programme looks for input files in the standard repository locations:

| Path | Description |
|------|-------------|
| `../infiles/<stem>.inp` | input configuration (same format as Fortran version) |
| `../outfiles/<stem>.out` | summary output (created automatically) |
| `../worksim/<stem>.npz` | focal-plane ntuple in NumPy compressed format |

Example:

```bash
python mc_single_arm.py hms_40deg_carbon_1560
```

## Input file format

The input file format is **identical** to the Fortran version.  Lines
beginning with `!` are comments.  The remaining non-blank lines are read in
the following order:

| # | Parameter | Units |
|---|-----------|-------|
| 1 | Number of Monte Carlo trials | — |
| 2 | Spectrometer (1=HMS, 2=SHMS) | — |
| 3 | Spectrometer momentum | MeV/c |
| 4 | Spectrometer angle | degrees |
| 5–6 | dp/p lower / upper limits | % |
| 7–8 | θ lower / upper limits | mrad |
| 9–10 | φ lower / upper limits | mrad |
| 11 | Horizontal beam size (full ±1σ width) | cm |
| 12 | Vertical beam size (full ±1σ width) | cm |
| 13 | Target length (or negative multifoil code) | cm |
| 14 | Fast-raster full width x | cm |
| 15 | Fast-raster full width y | cm |
| 16 | dp/p reconstruction cut | % |
| 17 | θ reconstruction cut | mrad |
| 18 | φ reconstruction cut | mrad |
| 19 | z reconstruction cut | cm |
| 20 | Radiation length of target | cm |
| 21 | Beam x offset (+x = left) | cm |
| 22 | Beam y offset (+y = up) | cm |
| 23 | Target z offset | cm |
| 24 | Spectrometer x offset | cm |
| 25 | Spectrometer y offset | cm |
| 26 | Spectrometer z offset | cm |
| 27 | Spectrometer xp offset | mrad |
| 28 | Spectrometer yp offset | mrad |
| 29 | Particle type (0=e, 1=p, 2=d, 3=π, 4=K) | — |
| 30 | Multiple scattering flag (0/1) | — |
| 31 | Wire-chamber smearing flag (0/1) | — |
| 32 | Store-all flag (0/1) | — |
| 33 | Beam energy for elastic calibration (optional, <0 = off) | MeV |
| 34 | Use sieve slit flag (optional, 0/1) | — |
| 35 | Target atomic number for elastic (optional) | — |

## Directory structure

```
python/
├── mc_single_arm.py   Main Monte Carlo programme (entry point)
├── constants.py       Physical constants
├── counters.py        Event counters (replaces Fortran COMMON blocks)
├── state.py           Track state singleton (replaces COMMON /track/)
├── target_cans.py     Target can geometry (cryocylinder, cryotuna, cryotarg2017)
├── shared/
│   ├── loren.py       Lorentz boost
│   ├── musc.py        Multiple scattering (Lynch–Dahl)
│   ├── project.py     Field-free drift with optional decay
│   ├── rng.py         Random number generators (Mersenne Twister, Box-Muller)
│   ├── rotations.py   Coordinate frame rotations (haxis, vaxis)
│   └── transp.py      COSY polynomial forward transport
├── hms/
│   ├── hut.py         HMS detector hut simulation
│   ├── mc_hms.py      HMS magnet transport (collimator → hut)
│   └── recon.py       HMS COSY reconstruction
└── shms/
    ├── hut.py         SHMS detector hut simulation
    ├── mc_shms.py     SHMS magnet transport (HB → hut)
    └── recon.py       SHMS COSY reconstruction
```

## Ntuple format

The ntuple is written as a NumPy `.npz` file containing an array named
`ntuple`.  Each row corresponds to one accepted (or all, if `store_all=1`)
event.  The columns mirror the Fortran HBOOK ntuple:

**HMS** (23 columns per row):

| Col | Variable |
|-----|----------|
| 0 | x_fp |
| 1 | y_fp |
| 2 | dx_fp |
| 3 | dy_fp |
| 4 | x_tgt_init |
| 5 | y_tgt_init |
| 6 | dph_init (rad) |
| 7 | dth_init (rad) |
| 8 | z_tgt_init |
| 9 | dpp_init (%) |
| 10 | y_tgt_recon |
| 11 | dph_recon (rad) |
| 12 | dth_recon (rad) |
| 13 | z_tgt_recon |
| 14 | dpp_recon (%) |
| 15 | fry |
| 16–19 | sieve placeholders |
| 20 | hSTOP_id |
| 21 | beam x |
| 22 | beam y |

**SHMS** columns follow the same layout with SHMS-specific quantities.

## Differences from the Fortran version

| Feature | Fortran | Python |
|---------|---------|--------|
| Ntuple format | HBOOK `.bin` | NumPy `.npz` |
| Random seed | `sgrnd(time8())` | `random.seed(time.time())` |
| COSY coefficient loader | Inline in each subroutine | `shared/transp.py` |
| Parallelism | None | None (single-threaded) |

The physics and acceptance geometry are identical to the Fortran original.
