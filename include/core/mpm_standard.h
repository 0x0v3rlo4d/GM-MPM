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


// include/core/mpm_standard.h

#pragma once

#include "mpm_base.h"
#include <vector>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

namespace gm_mpm {

class MPMStandard : public MPMBase {
public:
    MPMStandard(sycl::queue& queue);
    ~MPMStandard() override;

    void particleToGrid(sycl::queue& queue) override;
    void computeGridForces(sycl::queue& queue) override;
    void updateGridVelocities(sycl::queue& queue) override;
    void gridToParticle(sycl::queue& queue) override;
    void updateDeformationGradient(sycl::queue& queue) override;

    void setParticles(const std::vector<Particle>& particles);
    std::vector<Grid> getGrid() const;
    std::vector<Particle> getParticles() const;

protected:
    sycl::double3 interpolate(const sycl::double3& position, sycl::queue& queue) override;
    Matrix3d computeStress(const Matrix3d& deformationGradient, const Particle& particle) override;

    sycl::buffer<Particle, 1> d_particles;
    sycl::buffer<Grid, 1> d_grid;
};

} // namespace gm_mpm