
# Project Description

In quantum optics, a common difficulty is finding the right set of operations that produces a desired quantum state. In this project, we would like you to investigate whether a *neural network* (NN) can be trained to predict the appropriate *quantum circuit configurations* that generate a specified *target quantum state*.

<img width="897" height="4000" alt="image" src="https://github.com/user-attachments/assets/4289f558-45a5-4298-8f5d-f655570e050b" />

In quantum optics, we usually work with *continuous‑variable (CV) quantum information*. Here, information is encoded in observables with continuous spectra, such as the quadratures (position and momentum) of the electromagnetic field. This is in contrast to the more widely studied discrete two‑level systems. The general idea is to apply transformations to states of light and to use a phase‑space representation, such as the *Wigner function*, to visualise the resulting states. A Wigner function plays the role of a quasi‑probability distribution over position and momentum. It encodes the same information as the density matrix and reproduces the correct position and momentum statistics as marginals, but unlike a true classical probability distribution, it can take negative values. These negative regions signal non‑classical features of the state and are closely related to quantum interference.

In this field, we often make a distinction between **Gaussian** and **non‑Gaussian** states of light. Gaussian states and Gaussian operations are relatively easy to generate experimentally and are also efficiently classically simulable. However, non‑Gaussian states are required to enable universal quantum computation and to exhibit strong forms of non‑classicality, which act as key resources for universal CV quantum computing, quantum error correction, entanglement distillation, and possible quantum advantages.

Many common optical operations—such as beam splitters, phase shifters, and squeezing operations—preserve Gaussianity and therefore cannot produce non‑Gaussian states on their own. The most experimentally practical way of producing non‑Gaussian states is through non‑Gaussian measurements. In this project, we will focus on **photon‑number‑resolving detectors (PNRDs)**, where the measurement collapses part of an entangled state and the conditional resulting state becomes non‑Gaussian.

That said, finding the right sequence of operations to generate specific non‑Gaussian states is highly non‑trivial. For this reason, we believe that **NNs** could provide insight into discovering suitable circuit architectures capable of producing such states. To make the problem feasible, we suggest focusing on a circuit of the type depicted in the figure. Here, squeezed Gaussian states are used as inputs (with a \(\pi/2\) phase offset), followed by a set of parameterised beam splitters. The circuit (prior to photon detection) is [known to be capable of implementing any unitary transformation](https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460), provided that there are at least \(M(M-1)/2\) beam splitters, where \(M\) is the number of modes (horizontal lines in the circuit depiction). Each beam splitter has two parameters: theta, the beam splitting ratio, and phi, the relative phase. In addition, the final state is dependent on the squeezing parameters \(r_i\) of the input states and the number of photons detected by the detectors. 

With this in mind, we would like you to generate training data using https://github.com/XanaduAI/MrMustard and the provided code snippets. You will then build a neural network using PyTorch, TensorFlow, JAX, or your preferred machine‑learning library, and train it. As inputs to the NN, use the pixels of an image of the resulting Wigner function and as outputs the required parameters of the quantum circuit that produces the corresponding Wigner function. See the above image for a quick representation of the workflow. 

### Useful comments and bounds for parameters

- In order to make the simulation tractable, we recommend limiting the **max value for abs(\(r_i\)) to 0.8**. 
- In addition, the maximum number of photons detected in a single PNRD should be limited to 3.
- The mixing angle (theta) of a beam splitter is bounded between \(0\) and \(\pi/2\) (inclusive).
- The relative phase of a beam splitter is bounded between \(0\) and \(2\pi\) (exclusive).

The parameters theta and phi can, in principle, take on any real value, but values outside these ranges correspond to redundant parameterisations rather than new physical operations. You should therefore consider bounding the outputs of your neural network to these ranges. The squeezing values can also be complex, but with these specific alternating \(\pi/2\) input states along the \(x\) and \(p\) directions, we ensure \(r\) is real.

A useful reference for those interested in CV quantum information is the overview by *Jonatan Bohr* on the https://arxiv.org/abs/2102.05748 used in continuous‑variable quantum information.



### Getting started

1. Install the [uv](https://docs.astral.sh/uv/) package and project manager.
2. Fork the _qst-hack-2026_ repository and clone it to your local machine.
3. In the terminal, `cd` to the `Case 7` folder and run `uv sync` to set up a virtual environment according to the prescriptions in the `pyproject.toml` file.
4. If you want to work with notebooks: Run `uv run python -m ipykernel install --user --name qst-hack-2026-case7` to install that environment as a Jupyter kernel that can be selected in VS Code and Jupyter Lab.



Alternative setup, if you want to start from a clean folder:

2. Create an environment in a new folder:  
   ```
   uv init <projectname>
   cd <projectname>
   ```
3. Add [MrMustard (development version)](https://github.com/XanaduAI/MrMustard) to the environment:
   ```
   uv add "mrmustard @ git+https://github.com/XanaduAI/MrMustard.git"
4. Add plotly and jupyter:
   ```
   uv add plotly jupyter
   ```
5. Start Jupyter Lab from within the `projectname` environment,
   ```
   uv run jupyter lab
   ```
   or open the folder in VS Code.  
   You probably want to install the kernel as in point 4 above.

Start hacking!


### Contacts for Case 7
- Luis González Hofmann, lcgho@dtu.dk
- Timour Javar Magnier, tkjma@dtu.dk
- Jonas Neergaard-Nielsen, jsne@fysik.dtu.dk
