# PNNL — Pie's Neural Network Library

A small, dependency-free neural network library written in C, supporting
feedforward networks, backpropagation, and evolutionary mutation for
population-based training.

## Features

- Feedforward multi-layer networks with configurable layer sizes
- Four activation functions: ReLU, Linear, Sigmoid, Tanh — with matched
  weight initialization (He for ReLU, Xavier for the rest)
- Backpropagation via mean squared error, with a lower-level API
  (`network_backward_from_deltas`) for plugging in other loss/objective
  functions (e.g. reinforcement learning)
- Deep cloning and random mutation of networks, for evolutionary
  strategies / genetic algorithm style training
- Versioned binary save/load format, with corruption/mismatch checks
- No external dependencies beyond the C standard library and libm

## Project structure

```
.
├── Makefile
├── src/
│   ├── main.c       # Entry point / example usage
│   ├── neuralnet.c  # Core network, forward pass, backprop, mutation
│   └── storage.c    # Save/load to disk
└── lib/
    ├── neuralnet.h
    └── storage.h
```

## Building

```
make        # Builds bin/main
make run    # Builds and runs bin/main
make clean  # Removes build artifacts
```

Requires `gcc` and `make`. Linking against `libm` may be required on some
platforms (`Sigmoid`/`Tanh` use `expf`/`tanhf`); add `-lm` to the link step
in the Makefile if your toolchain doesn't include it automatically.

## Basic usage

```c
#include "neuralnet.h"

// Build a network: 3 inputs -> 5 hidden (ReLU) -> 2 outputs (linear)
Layer layers[2];
layers[0] = create_layer(3, 5, ACTIVATION_RELU);
layers[1] = create_layer(5, 2, ACTIVATION_LINEAR);
Network net = create_network(2, layers);

// Forward pass
float input[3] = {1.0f, 0.5f, -0.2f};
forward(&net, input);
// net.layers[net.num_layers - 1].outputs now holds the result

// Training (supervised, MSE-based)
TrainingContext ctx = create_training_context(&net);
float target[2] = {1.0f, 0.0f};
network_backward(&net, &ctx, input, target, 0.01f);
free_training_context(&ctx, net.num_layers);

free_network(&net);
```

## Saving and loading

```c
#include "storage.h"

save_network(&net, "model.bin");
Network loaded = load_network("model.bin");
```

`load_network` returns an empty network (`num_layers == 0`, `layers == NULL`)
on any failure — check with `network_is_valid(&net)` before using the result.
Save files are versioned (`FORMAT_VERSION` in `storage.h`); loading a file
saved by an incompatible version fails cleanly rather than misreading data.
See `Storage Changelog.md` for the format's version history.

## Evolutionary training

For non-gradient-based training (e.g. evolutionary strategies), networks can
be cloned and randomly perturbed instead of backpropagated:

```c
Network offspring = clone_network(&parent);
mutate_network(&offspring, 0.1f, 0.05f);  // mutation_rate, mutation_strength
```

`clone_network` performs a full deep copy — mutating a clone never affects
the original.

## Status

The core library (forward pass, backprop, save/load, mutation) is complete
and tested against MNIST digit classification. Ongoing work is on building
an environment and population-management loop for evolutionary training of
agents/creatures — that layer doesn't exist yet and isn't part of this
library itself.