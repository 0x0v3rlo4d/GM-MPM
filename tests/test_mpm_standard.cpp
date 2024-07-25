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

// test/test_mpm_standard.cpp

#include <gtest/gtest.h>
#include "mpm_standard.h"
#include <CL/sycl.hpp>
#include <vector>

namespace gm_mpm {
namespace test {

class MPMStandardTest : public ::testing::Test {
protected:
    sycl::queue queue;
    MPMStandard mpm;

    MPMStandardTest() : queue(sycl::default_selector()), mpm(queue) {}

    void SetUp() override {
        // Initialize test scenario
        // Set up a simple block of particles
        std::vector<Particle> particles(1000);
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].position = sycl::double3(
                (i % 10) * 0.1,
                ((i / 10) % 10) * 0.1,
                (i / 100) * 0.1
            );
            particles[i].velocity = sycl::double3(0.0, 0.0, 0.0);
            particles[i].mass = 1.0;
            particles[i].volume = 0.001;
            // Initialize other particle properties as needed
        }

        // Update the MPM system with these particles
        mpm.setParticles(particles);
    }
};

TEST_F(MPMStandardTest, ParticleToGridTransfer) {
    mpm.particleToGrid(queue);
    
    // Check if mass and momentum were transferred correctly
    std::vector<Grid> grid = mpm.getGrid();
    double totalMass = 0.0;
    sycl::double3 totalMomentum(0.0, 0.0, 0.0);
    
    for (const auto& cell : grid) {
        totalMass += cell.mass;
        totalMomentum += cell.momentum;
    }
    
    EXPECT_NEAR(totalMass, 1000.0, 1e-6);
    EXPECT_NEAR(totalMomentum.x(), 0.0, 1e-6);
    EXPECT_NEAR(totalMomentum.y(), 0.0, 1e-6);
    EXPECT_NEAR(totalMomentum.z(), 0.0, 1e-6);
}

TEST_F(MPMStandardTest, FullTimeStep) {
    // Run a full time step
    mpm.particleToGrid(queue);
    mpm.computeGridForces(queue);
    mpm.updateGridVelocities(queue);
    mpm.gridToParticle(queue);
    mpm.updateDeformationGradient(queue);
    
    // Check if particles have moved due to gravity
    std::vector<Particle> particles = mpm.getParticles();
    for (const auto& particle : particles) {
        EXPECT_LT(particle.position.y(), 0.0);
        EXPECT_LT(particle.velocity.y(), 0.0);
    }
}

TEST_F(MPMStandardTest, Conservation) {
    // Run several time steps
    for (int i = 0; i < 10; ++i) {
        mpm.particleToGrid(queue);
        mpm.computeGridForces(queue);
        mpm.updateGridVelocities(queue);
        mpm.gridToParticle(queue);
        mpm.updateDeformationGradient(queue);
    }
    
    // Check conservation of mass and momentum
    std::vector<Particle> particles = mpm.getParticles();
    double totalMass = 0.0;
    sycl::double3 totalMomentum(0.0, 0.0, 0.0);
    
    for (const auto& particle : particles) {
        totalMass += particle.mass;
        totalMomentum += particle.mass * particle.velocity;
    }
    
    EXPECT_NEAR(totalMass, 1000.0, 1e-6);
    EXPECT_NEAR(totalMomentum.x(), 0.0, 1e-6);
    EXPECT_NEAR(totalMomentum.z(), 0.0, 1e-6);
    // Y-momentum will change due to gravity
    EXPECT_LT(totalMomentum.y(), 0.0);
}

} // namespace test
} // namespace gm_mpm

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}