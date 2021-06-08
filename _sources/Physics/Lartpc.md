
# Liquid Argon Time Projection Chamber (aka LArTPC)

Where you can learn more about the type of detector that creates the images we are trying to analyze
with `lartpc_mlreco3d`.

## The detector
### What LArTPC stands for
LArTPC is not the name of a new cryptocurrency (it's too hard to pronounce for that),
it stands for *Liquid Argon Time Projection Chamber*. It is a type of particle detector
that provides us with high resolution 2D or 3D images of charged particle trajectories.

```{figure} https://temigo.github.io/static/MicroBooNE3DEvent-b69f63c7bfc599f9e6fb16743c5e09d5-79df8.jpg
---
height: 300px
name: directive-fig
---
Example of LArTPC 3D reconstructed image from MicroBooNE experiment
```


### What it looks like
The detector is a huge tank filled with liquid argon (LAr) with a `uniform electric field`, typically 500 V/cm. Huge can mean 480 tons
(ICARUS detector, running now) or as much as  40,000 tons (DUNE detector, coming up) of liquid argon. This LAr volume is sandwiched by an anode and cathode where _very_ high voltage is applied (hundreds of kilovolt to produce 500 V/cm field across meters!).

In practice, it works like a modern bubble chamber. Charged particles - whether coming from cosmic rays, the accelerator beam, etc - travel through  the detector volume. Some of them also interact with the argon atoms, sending new particles on their way. 

```{figure} https://sites.slac.stanford.edu/neutrino/sites/neutrino.sites.slac.stanford.edu/files/styles/fix_height_400/public/icarus_cryostat.png?itok=39f-0Swu
---
height: 300px
name: directive-fig
---
That is only half of the ICARUS detector in Fermilab
```

```{figure} https://icarus.fnal.gov/wp-content/uploads/2018/04/ICARUS-PR-7--1440x492.jpg
---
height: 300px
name: directive-fig
---
This is what ICARUS looks like inside (without the liquid argon).
```

### A tale of two signals
When a charged particle travels through the LAr volume with a uniform electric field, two signals are produced:


1. it causes `ionization` along its path, creating Argon ion and ionization electrons. In the presence of a strong electric field, they cannot re-combine and instead start drifting toward the opposite direction. Positively charged Ar ions drift toward a cathod while negatively charged ionization electrons drift toward an anode. A charge-sensitive detector device is placed at the anode plane to collect those ionization electrons. While drifting, which typically takes a few milliseconds for a few meters, all ionization electrons travel toward the anode at the same speed (about 1400 m/s under the electric field 500 V/cm), retaining the shape of the source particle trajectory. The detector at the anode is either a `wire` or `pixel` based detector. A wire-based detector employ 3 planes of parallel wires, each plane at a different wire angle, and records 2D projection image of particle trajectories per plane. All existing and many planned LArTPCs belong to this type. A pixel-based detector employs single plane consists of many small (typically square) pad of sensors instead of wires, and it can record 3D image of particle trajectories. This technology is recently developed, and is planned for one of detectors in the DUNE experiment.


2. `scintillation` lights are produced by de-excitation of Argon ions. There are two marjor paths for this de-excitation, and two time constants associated with them. About 23% of the lights are produced within a few nano-seconds while the rest is produced over the course of micro-seconds. Once produced, these photons are detected by optical detectors within a few to several nano-seconds depending on the detector geometry. One notable point about LAr is, unlike liquid scintillator or water, it is transparent to its own scintillation light. So these scintillation photons can travel straight to optical detectors without absorption and re-emission. However, LAr scintillation photons have the wavelength that is too short for typical optical detectors (e.g. Silicon or ordinary Photomultiplier Tubes, or `PMTs`). A wavelength shifter such as `TPB` is used to coat the detector surface so that these scintillation photons can be detected efficiently.


```{margin} In summary
LArTPC detectors are modern bubble chambers (filled with liquid argon) that
measure both charge and light signals to reconstruct images of charged particle trajectories.
```

```{figure} ./lartpc.jpg
---
height: 600px
name: directive-fig
---
Schematic of a LArTPC detector, in this example ICARUS. Credit: Adrian Cho, Science. 10.1126/science.365.6453.532
```

### What do you do with charge and light signals?
The charge signal provides us an `image of charged particle trajectories`. Two key information in an image data are `spatial` (inferred depending on which wire or pixel it hits) and `calorimetric` (how many electrons, i.e. how much charge did we read out). This is the main signal for LArTPCs and the detector design is optimized for the charge signal. So far `lartpc_mlreco3d` has been focused on this part of the story.

Optical detectors in a LArTPC is typicaly much more sparse than the charge detector. For instance, in ICARUS, there are about 60,000 wires for detecting ionization electrons for imaging while there are only 180 PMTs to detect scintillation light. So the light signal is much more coarse for reconstructing the spatial and energy features of particles (though possible and useful still!). Yet, the optical detectors provide a `precise timing information` because scintillation photons are detected within nano-seconds after an event occurs inside the detector. This is much faster compared to the TPC signal, which can take milliseconds before detection due to drift of ionization electrons. 


## What does it have to do with neutrinos?
In neutrino oscillation experiments, we are counting neutrinos at both a near- and far-detector to see how many of them have changed a flavor (i.e. "oscillated"). Using this information and the energy of neutrinos, we can infer the neutrino oscillation parameters and compare if there is any difference in the behavior of neutrinos from anti-neutrinos (called `CP violation`).

When neutrinos interact with Argon nucleus, they produce charged particles as products such as a proton, a muon, an electron, charged pions, or maybe sometimes even kaons! LArTPCs allow us to record full details of these charged particles in order to infer the flavor and energy of neutrinos. This, in principle, allows us to perform a high precision oscillation measurement. 


In order to reverse-engineer the neutrino flavor and energy from recorded particle trajectories, we have to `reconstruct` the types and kinematics of them first: that means identify individual particle trajectory in an image, determine the particle type and momentum, and group particles that originated from the same interaction (i.e. one image could contain many independent source of particles, or interactions).  This turns out to be a challenging task, and what `lartpc_mlreco3d` is designed to address as a generic reconstruction tool.

## Misc. questions
### Why argon?
Excellent question! Argon is widely available (most abundant noble gas on Earth), which means it's cheap! This is an important property for a scalable detector technology. Furthermore, it is relatively easy to handle (e.g. can be cooled with liquid Nitrogen, a cheap, widely available coolant) and has some nice physics properties (e.g. it produces lots of scintillation photons, about 24,000 photons per MeV at 500 V/cm). Read more [here](https://en.wikipedia.org/wiki/Noble_gas#Occurrence_and_production).

### Why liquid rather than gas?
The cross-section of particle interactions scales with the density of the material. The density of gaseous Ar is a thousandth of liquid Ar (see https://lar.bnl.gov/properties/). Hence liquid argon is much more effective to observe particle interactions!

### Why does argon purity matter?
The higher the liquid argon purity is, the less likely the ionization electrons (which are free electrons
as they drift) are to be captured by `impurities` (e.g. Oxygen) before they reach the charge-sensitive detector elements. Similarly, scintillation photons suffer from an absorption by impurities (e.g. Nitrogen). The larger the volume, the more crucial to remove impurities because of the longer distance ionization electrons and scintillation photons must travel to the detector elements (hence the greater chance of being absorbed). A purification system, which typically consists of a circulation pump and a filters to remove impurities, can be used to reduce the level of impurities. A continuous monitoring of the purity is also important to avoid situations and to apply calibrations to adjust for the lost signal due to impurities.



## Neutrino LArTPC experiments in our group

- MicroBooNE (not running anymore, still analyzing data), 87 ton of LAr
- ICARUS (480 tons) + SBND (112 tons) = SBN program (currently running)
- DUNE-ND + DUNE-FD = DUNE program (flagship LArtPC neutrino experiment in the US, under construction, will be 40,000 tons of LAr)

You might also hear about ArgonCube (modular prototype of LArTPC for DUNE-ND) and ProtoDUNE (R&D prototype for DUNE).

## Glossary

```{glossary}

ghost point
  A 3D point not meant to be
  
wire plane
  A set of wires parallel to each other, all in the same geometrical plane.
  
time projection chamber (TPC)
  Detector that images a particle 3D trajectory using electromagnetic fields and a volume of sensitive liquid/gas.
```