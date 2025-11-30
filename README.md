536376 Marek Eibel, 536601 Jakub Čížek

# NewNeuralNetwork 1.0.0

A straight-forward MLP training framework implemented in C++. 

https://github.com/Spaghetti-Code-Studio/NewNeuralNetwork

## Running the project

The build process is managed by **CMake** and has been tested for both **MSVC** and **GCC**. The project can be built and ran in multiple configurations:
- **Debug** `./scripts/run-debug` with tests enabled and no optimization.
- **Release** `./scripts/run-release.sh` with tests enabled and optimization flags.
- **Production** `./scripts/run-production.sh` without tests and with optimization flags.

The **production** configuration should be used on the Aura server as is also the configuration used in `run.sh`. **Release** cannot be used there since its expects the paths of all files to be relative to the executable in `/build`. Also compilation of Catch2 for tests severely slows the build process.

### Optional external dependencies:
While the project implements all necessary functionality, the project optionally requires [Catch2](https://github.com/catchorg/Catch2) if testing is enabled, and also **OpenMP** for parallel computation of matrix multiplication. Neither is required for the **Production** configuration, although without **OpenMP**, the the execution time will be considerably longer. 

Both dependencies are downloaded automatically by CMake during the build process.

### Internally used libraries

This project uses two libraries located in `/vendors` which are unrelated to machine learning. \
Both are under the MIT license:

- [Result monadic type](https://github.com/bitwizeshift/result) by Matthew Rodusek
- [JSON parser](https://github.com/nlohmann/json) by Niels Lohmann

## Training setup & Performance

Most hyperparameters can be tuned by changing the `config.json` file located in the root directory of the project. The current settings achieves **over 89% accuracy** with the set seed, but was also tested to achieve +88% accuracy on all other allowed seeds. The training can be expected to take around 3-5 minutes on Aura with 8 parallel threads.