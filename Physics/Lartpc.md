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
The detector is a huge tank filled with liquid argon. Huge can mean 700 tons
(ICARUS detector, running now) or as much as  70,000 tons (DUNE detector, coming up) of liquid argon.
It works like a modern bubble chamber. Particles - whether coming from cosmic rays, the accelerator beam, etc - travel through 
the detector volume. Some of them also interact with the argon atoms, sending new particles 
on their way. 

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
Whenever a charged particle travels through the LArTPC detector, it ionizes the argon atoms on its way.
This has 2 consequences: 
1. it knocks off some ionization electrons that all drift (slowly) in the same direction because of an electric field
that we create inside the detector. These electrons will eventually meet either wires (if 2D detector, organized in 3 planes of parallel wires, each plane at a different angle, at the edges of the detector) or "pixels" (if 3D, new detector, placed at regular spacing within the volume) sensors.
2. as they lose energy these ionization electrons create scintillation photons which travel fast (speed of light!)
and are detected by an array of PMTs (photomultipliers) on the walls of the detector.

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
The charge signal provides us with two informations: spatial (inferred depending on which wire or pixel it hits) and quantitative (how many electrons, i.e. how much charge did we read out). We can reconstruct a lot of information about the particles based on this. So far `lartpc_mlreco3d` has been focused on this part of the story.

Light signal is also processed and organized into *optical flashes* which should match (in spatial location and intensity) with the reconstructed particles in the TPC volume. For example, it can help to weed out cosmic rays that happen outside of the beam time window (no chance that
it could be a neutrino). 

## What does it have to do with neutrinos?
We can use a LArTPC to study cosmic rays, but this would not be very exciting.
Things become interesting when neutrinos cross the detector volume. Neutrinos have no charge, so we
cannot see them directly. Bummer. Neutrinos however might knock off a proton or a neutron
from an argon atom. Protons have a positive charge, so we will see them - both in charge and light.
Neutrons will only be seen through the light signal and the potential secondary protons that they
knock off (if they have enough energy).

In long baseline neutrino oscillation experiments, we are counting neutrinos flavors and energy 
at both a near- and far-detector to see if some neutrinos have changed flavor as they were traveling,
and for example to measure the neutrino oscillation parameters with high precision. This means that there are two 
fundamental quantities that we are trying to infer about the neutrinos that travel
through our LArTPC detectors: the __neutrino flavor and energy__. 

We have to *reconstruct* various intermediate particles: that means identify which energy deposits belong to each of them, 
what kind of particle they were and what kinematics they had, where they interacted with each other, etc. 
That allows further downstream physics analysis (not included in `lartpc_mlreco3d` but hopefully greatly simplified by its output) in order
to infer these two key properties (flavor and energy) about the original neutrino that was wandering through our LArTPC and luckily set
off these chains of interactions.

## Misc. questions
### Why argon?
Excellent question! Argon is widely available (most abundant noble gas on Earth) and its liquid phase has excellent scintillation properties (amount of light emitted per unit of energy deposited by the ionizing particle ~ 40,000 photons per MeV). Also, it is cheap and we need loads of it. (See https://en.wikipedia.org/wiki/Noble_gas#Occurrence_and_production)

### Why liquid rather than gas?
The cross-section of particle interactions scales with the density of the material. The density of gaseous Ar is a thousandth of liquid Ar (see https://lar.bnl.gov/properties/). Hence liquid argon is much more effective to observe particle interactions!

### Why does argon purity matter?
The higher the liquid argon purity is, the less likely the ionization electrons (which are free electrons
as they drift) are to be captured by an argon atom before they reach the wire planes.


## Other fun facts about LArTPCs
Ask Kazu about it. He has a whole list.

## Neutrino LArTPC experiments in our group
You might hear about them quite often if you work with us. As a quick overview to clear out the confusion,
without going into the details of the physics they are each looking for and measuring. The main experiments
that we are involved into include (in increasing LAr volume and historical order):

- MicroBooNE (not running anymore, still analyzing data), 170 ton of LAr
- ICARUS (760 tons) + SBND (112 tons) = SBN program (currently running)
- DUNE-ND + DUNE-FD = DUNE program (flagship LArtPC neutrino experiment in the US, under construction, will be 70,000 tons of LAr)

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