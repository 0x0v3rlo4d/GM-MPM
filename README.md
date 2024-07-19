# GM-MPM
**Generalized Modular Material Point Method** (GM-MPM) is a material point method graphics simulator adapted to various heteregenous system using SYCL implementation under AdaptiveCpp compilation platform. This enables GM-MPM to run on all supported hardware and lets you modify, build, and run custom features to GM-MPM.

It natively supports voxelization, fluid simulation, and rigid body simulation within one program. For the time being, GM-MPM only supports `*.obj` file to be simulated within, more 3D formats will follow.

It supports logging capabilities for physics factors such as internal force, velocity, stress, shear, temperature, pressure, and friction, and performance factors such as entire simulation time, iteration time, and frame-per-second (FPS) rate.

## Installation
### Operating System Support
Currently **GM-MPM is** meant to be run on **Linux only** due to dependencies with AdaptiveCpp compiler. However, GM-MPM is able to run on most Linux-based heteregenous systems with varying hardware configuration that are supported by AdaptiveCpp such as: LLVM, NVIDIA CUDA, AMD ROCm, SPIR-V/Level Zero, and SPIR-V/OpenCL.

### Prerequisites
To build and install GM-NPN you need the following:
1. CMake (`cmake`)
2. [AdaptiveCpp]([https://](https://github.com/AdaptiveCpp/AdaptiveCpp)), follow the [installation]([https://](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md)) there according your system's hardware configuration. For example, for CUDA-enabled systems you might want to do the following:

```bash
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DWITH_CUDA_BACKEND=ON
make install
``` 
For any question regarding AdaptiveCpp installation, you should go to their Discord server.

### Installation
After getting AdaptiveCpp and `cmake`

```md
# WORK IN PROGRESS. PLEASE WAIT. . . 
```


## About the Project
GM-MPM's development is currently primarily led by Computation and Algorithm Laboratory at Department of Computer Science and Electronics of Universitas Gadjah Mada, with supports and contributions from a growing community. Our goal is to create an MPM simulation program that is platform-agnostic and could be used from personal computer to high-performance computing machines with varying heterogenous architecture settings.

### Get in Touch
Join us on [Discord](https://discord.gg/XemFJ4Taru) or open a discussion or issue in this repository.

### Contributing to GM-MPM
Contributing to GM-MPM is always open, we are looking up for your pull request!

We are very pleased if your organization or institution wants to support the development of GM-MPM in certain official capacity, we are open to discuss and develop the community. Please reach out to us!

### Acknowledgement
We gratefully thank contributors from the community. We thank Department of Computer Science and Electronics of Universitas Gadjah Mada for their support on this project and countless people there that made this project possible. We also thank Aksel Alpay et.al. and AdaptiveCpp community for facilitating GM-MPM to run across varying supported heterogenous systems.

### Cited Works
* Aksel Alpay and Vincent Heuveline. 2023. *One Pass to Bind Them: The First Single-Pass SYCL Compiler with Unified Code Representation Across Backends.* In Proceedings of the 2023 International Workshop on OpenCL (IWOCL '23). Association for Computing Machinery, New York, NY, USA, Article 7, 1–12. https://doi.org/10.1145/3585341.35853513
* Pirouz Nourian, Ken Arroyo Ahori, and Romulo Goncalves. 2015. *geospatial-voxels*. https://github.com/NLeSC/geospatial-voxels/