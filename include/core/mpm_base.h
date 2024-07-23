/*
 * This file is part of GM-MPM, a material point method graphics simulator 
 * adapted to various heteregenous system using SYCL implementation 
 * under AdaptiveCpp compilation platform.
 * 
 * 
 * Copyright GM-MPM Contributors
 *
 * GM-MPM is released under the MIT License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: MIT


// include/core/mpm_base.h

#pragma once

#include <vector>
#include <cstdint>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

namespace gm_mpm {

// Custom 3x3 matrix type
struct Matrix3d {
    double data[9];  // Row-major order

    Matrix3d() {
        for (int i = 0; i < 9; ++i) data[i] = 0.0;
    }

    double& operator()(int row, int col) { return data[row * 3 + col]; }
    const double& operator()(int row, int col) const { return data[row * 3 + col]; }
};

struct Particle {
    sycl::double3 position;
    sycl::double3 velocity;
    Matrix3d deformationGradient;
    double mass;
    
    // Additional properties
    Matrix3d stress;
    Matrix3d strain;
    double temperature;
    double pressure;
    sycl::double3 internalForce;
    double friction;
    
    // Material properties
    double youngsModulus;
    double poissonRatio;
    double yieldStress;
    
    // Thermal properties
    double heatCapacity;
    double thermalConductivity;
    
    double damageParameter;
};

struct Grid {
    sycl::double3 velocity;
    sycl::double3 force;
    double mass;
    
    sycl::double3 momentum;
    Matrix3d stress;
    double temperature;
    
    int materialID;
    
    bool isBoundary;
    sycl::double3 boundaryNormal;
};

class MPMBase {
public:
    virtual ~MPMBase() = default;

    virtual void particleToGrid(sycl::queue& queue) = 0;
    virtual void computeGridForces(sycl::queue& queue) = 0;
    virtual void updateGridVelocities(sycl::queue& queue) = 0;
    virtual void gridToParticle(sycl::queue& queue) = 0;
    virtual void updateDeformationGradient(sycl::queue& queue) = 0;

protected:
    std::vector<Particle> particles;
    std::vector<Grid> grid;
    
    double timeStep;
    sycl::double3 gravity;
    sycl::int3 gridResolution;
    double gridSpacing;

    double bulkModulus;
    double shearModulus;

    double totalTime;
    double currentTime;
    int currentStep;

    std::vector<sycl::int3> boundaryNodes;

    virtual sycl::double3 interpolate(const sycl::double3& position, sycl::queue& queue) = 0;
    virtual Matrix3d computeStress(const Matrix3d& deformationGradient, const Particle& particle) = 0;
};

} // namespace gm_mpm