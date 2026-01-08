## Reference repos & libraries (to avoid reinventing the wheel)

These are good starting points to **fork, read, or borrow patterns from**. Favor libraries with active maintenance and clear docs.

### Spiking neural networks (SNN)

- **snnTorch**: PyTorch-first spiking layers and training utilities  
  Repo: `https://github.com/jeshraghian/snntorch`
- **Norse**: spiking modules for PyTorch, good for research codebases  
  Repo: `https://github.com/norse/norse`
- **Brian2**: neuroscience simulator (great for STDP/biophysical realism)  
  Repo: `https://github.com/brian-team/brian2`
- **Lava** (Intel): neuromorphic + event-based workflows  
  Repo: `https://github.com/lava-nc/lava`

### Hebbian / local learning rules (Oja, BCM, anti-Hebbian, etc.)

- **SoftHebb** (paper + code; modern Hebbian feature learning)  
  Repo: `https://github.com/NeuromorphicComputing/softhebb`
- **BindsNET** (spiking + STDP examples; some code patterns are useful even if you don’t adopt it)  
  Repo: `https://github.com/BindsNET/bindsnet`

### Predictive coding / analysis-by-synthesis

- **PredNet** (classic predictive coding-style architecture reference)  
  Repo: `https://github.com/coxlab/prednet`
- **PCN / predictive coding networks** (search term to explore variants):  
  GitHub search: `predictive coding network pytorch`

### Cognitive architectures / graph-based memory

- **Nengo** (neurally plausible cognitive modeling toolkit; SPA is especially relevant)  
  Repo: `https://github.com/nengo/nengo`
- **ACT-R** (classic cog-arch; good conceptual reference, not necessarily code reuse)  
  Site: `https://act-r.psy.cmu.edu/`

### Tracking & perception plumbing

- **Ultralytics** (YOLOv8; for feature extraction hooks and output formats)  
  Repo: `https://github.com/ultralytics/ultralytics`
- **DeepSORT** (tracking-by-detection; useful baseline before learning recurrence)  
  Search term: `DeepSORT pytorch`

### Practical advice on reuse

- Borrow **interfaces** (how state is stepped, how logs/metrics are produced) more than full models.
- Keep pretrained perception (YOLO/backbones) **frozen** initially; use it as a stable sensory substrate.
- Only introduce SNN recurrence after you can show non-spiking baselines + ablations.

