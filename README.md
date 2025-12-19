# NewNeuralNetwork 1.2.0

A straight-forward MLP training framework implemented in C++.

## Running the project

The build process is managed by **CMake** and has been tested for both **MSVC** and **GCC**. The project can be built and ran in multiple configurations:
- **Debug** `./scripts/run-debug` with tests enabled and no optimization.
- **Release** `./scripts/run-release.sh` with tests enabled and optimization flags.
- **Production** `./scripts/run-production.sh` without tests and with optimization flags.

The **production** configuration should be used on the Aura server and is also the configuration used in `run.sh`. **Release** cannot be used there since its expects the paths of all files to be relative to the executable in `/build`. Also compilation of tests severely slows the build process.

### Optional external dependencies:
While the project implements all necessary functionality, the project optionally requires [Catch2](https://github.com/catchorg/Catch2) if testing is enabled, and also **OpenMP** for parallel computation of matrix multiplication. Neither is required for the **Production** configuration, although without **OpenMP**, the the execution time will be considerably longer. 

Both dependencies are downloaded automatically by CMake during the build process.

### Used 3rd party libraries

This project uses two libraries located in `/vendors` which are unrelated to machine learning. \
Both are under the MIT license:

- [Result monadic type](https://github.com/bitwizeshift/result) by Matthew Rodusek
- [JSON parser](https://github.com/nlohmann/json) by Niels Lohmann

## Training setup & Performance

Most hyperparameters can be tuned by changing the `config.json` file located in the root directory of the project. The current settings achieves **89.5% accuracy** with the set seed, but was also tested to achieve +88% accuracy on all other allowed seeds. The training can be expected to take around 1-2 minutes on Aura with 16 parallel threads (if necessary, this can be decreased to 3 while still satisfying the time constrains, this can be done with `"hardThreadsLimit": 3`).

The chosen network configuration uses 4 layers: 3 hidden layers (leaky ReLU) and softmax output layer.

## Architecture

We tried to keep the implementation general and modular. The core of the MLP abstraction is the `ILayer` interface, the network can be constructed by gradually adding `ILayer` objects and is finalized by adding a `IOutputLayer` object.

### Implemented optimizations:
- momentum
- weight decay
