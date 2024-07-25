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


// src/core/mpm_standard.cpp

#include "mpm_standard.h"
#include <CL/sycl.hpp>
#include <cmath>

namespace sycl = cl::sycl;

namespace gm_mpm {

inline double custom_cbrt(double x) {
    if (x == 0) return 0;
    
    double sign = (x > 0) ? 1 : -1;
    x = std::abs(x);
    
    double result = x;
    double delta;
    for (int i = 0; i < 8; i++) { // 8 iterations should be enough for double precision
        delta = (result * result * result - x) / (3 * result * result);
        result -= delta;
    }
    
    return sign * result;
}

    const std::vector<Particle>& MPMStandard::getParticles() const {
        return particles;
    }

    const std::vector<Grid>& MPMStandard::getGrid() const {
        return grid;
    }

MPMStandard::MPMStandard(sycl::queue& queue)
    :// Initialize standard MPM
    d_particles(sycl::range<1>(1000)),
    d_grid(sycl::range<1>(100 * 100 * 100))
{
    timeStep = 0.001;
    gravity = sycl::double3(0.0, -9.81, 0.0);
    gridSpacing = 0.1;
    gridResolution = sycl::int3(100, 100, 100);

    bulkModulus = 1e6;
    shearModulus = 5e5;

    totalTime = 10.0;
    currentTime = 0.0;
    currentStep = 0;

    queue.submit([&](sycl::handler& cgh) {
        auto particles_acc = d_particles.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class init_particles>(sycl::range<1>(1000), [=](sycl::id<1> idx) {
            particles_acc[idx].position = sycl::double3(0.0, 0.0, 0.0);
            particles_acc[idx].velocity = sycl::double3(0.0, 0.0, 0.0);
            particles_acc[idx].mass = 1.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    particles_acc[idx].deformationGradient(i, j) = (i == j) ? 1.0 : 0.0;
                }
            }
        });
    });
}

MPMStandard::~MPMStandard() {
    // Cleanup if necessary
}

void MPMStandard::setParticles(const std::vector<Particle>& newParticles) {
    particles = newParticles;
    
    // Resize the device buffer if necessary
    if (d_particles.get_count() != particles.size()) {
        d_particles = sycl::buffer<Particle>(particles.data(), sycl::range<1>(particles.size()));
    } else {
        // Update the existing buffer
        queue.submit([&](sycl::handler& cgh) {
            auto acc = d_particles.get_access<sycl::access::mode::write>(cgh);
            cgh.copy(particles.data(), acc);
        });
        queue.wait();
    };
}

void MPMStandard::particleToGrid(sycl::queue& queue) {
    // Calculate total mass before transfer on the host
    double totalMassBefore = std::accumulate(particles.begin(), particles.end(), 0.0,
        [](double sum, const Particle& p) { return sum + p.mass; });
    std::cout << "Total mass before transfer: " << totalMassBefore << std::endl;

    // Reset grid mass and momentum
    std::fill(grid.begin(), grid.end(), Grid{});

    queue.submit([&](sycl::handler& cgh) {
        auto particles_acc = d_particles.get_access<sycl::access::mode::read>(cgh);
        auto grid_acc = d_grid.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(d_particles.get_count()), [=](sycl::id<1> idx) {
            const auto& particle = particles_acc[idx];
            sycl::double3 baseNode = sycl::double3(particle.position / gridSpacing);

            for (int i = 0; i < 8; ++i) {
                sycl::double3 node = baseNode + sycl::double3((i & 1), ((i & 2) >> 1), ((i & 4) >> 2));
                sycl::double3 nodePos = sycl::double3(node) * gridSpacing;
                sycl::double3 diff = (particle.position - nodePos) / gridSpacing;

                double weight = (1 - diff.x()) * (1 - diff.y()) * (1 - diff.z());

                size_t nodeIdx = node.x() + node.y() * gridResolution.x() + node.z() * gridResolution.x() * gridResolution.y();
                auto& cell = grid_acc[nodeIdx];

                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device> mass_ref(cell.mass);
                mass_ref.fetch_add(particle.mass * weight);

                auto momentum_x_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.momentum[0]);
                auto momentum_y_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.momentum[1]);
                auto momentum_z_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.momentum[2]);

                auto mass_times_velocity = particle.mass * particle.velocity * weight;
                momentum_x_ref.fetch_add(mass_times_velocity.x());
                momentum_y_ref.fetch_add(mass_times_velocity.y());
                momentum_z_ref.fetch_add(mass_times_velocity.z());
            }
        });
    });
    queue.wait();

    // Calculate total mass after transfer on the host
    double totalMassAfter = std::accumulate(grid.begin(), grid.end(), 0.0,
        [](double sum, const Grid& cell) { return sum + cell.mass; });
    std::cout << "Total mass after transfer: " << totalMassAfter << std::endl;
}

void MPMStandard::computeGridForces(sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        auto particles_acc = d_particles.get_access<sycl::access::mode::read>(cgh);
        auto grid_acc = d_grid.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(d_particles.get_count()), [=](sycl::id<1> idx) {
            const auto& particle = particles_acc[idx];
            Matrix3d stress = computeStress(particle.deformationGradient, particle);

            sycl::double3 baseNode = particle.position / gridSpacing;

            for (int i = 0; i < 8; ++i) {
                sycl::double3 node = baseNode + sycl::double3((i & 1), ((i & 2) >> 1), ((i & 4) >> 2));
                sycl::double3 nodePos = node * gridSpacing;
                sycl::double3 diff = (particle.position - nodePos) / gridSpacing;

                sycl::double3 weight_grad(
                    (i & 1) ? 1 : -1,
                    ((i & 2) >> 1) ? 1 : -1,
                    ((i & 4) >> 2) ? 1 : -1
                );
                weight_grad /= gridSpacing;

                // Compute force update
                sycl::double3 force_update;
                force_update.x() = -(stress(0,0) * weight_grad.x() + stress(0,1) * weight_grad.y() + stress(0,2) * weight_grad.z());
                force_update.y() = -(stress(1,0) * weight_grad.x() + stress(1,1) * weight_grad.y() + stress(1,2) * weight_grad.z());
                force_update.z() = -(stress(2,0) * weight_grad.x() + stress(2,1) * weight_grad.y() + stress(2,2) * weight_grad.z());
                force_update *= particle.volume;

                size_t nodeIdx = node.x() + node.y() * gridResolution.x() + node.z() * gridResolution.x() * gridResolution.y();
                auto& cell = grid_acc[nodeIdx];

                auto force_x_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.force[0]);
                auto force_y_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.force[1]);
                auto force_z_ref = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>(cell.force[2]);

                force_x_ref.fetch_add(force_update.x());
                force_y_ref.fetch_add(force_update.y());
                force_z_ref.fetch_add(force_update.z());
            }
        });
    });
}

void MPMStandard::updateGridVelocities(sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        auto grid_acc = d_grid.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(d_grid.get_count()), [=](sycl::id<1> idx) {
            auto& cell = grid_acc[idx];
            if (cell.mass > 0) {
                cell.velocity = cell.momentum / cell.mass;
                cell.velocity += (cell.force / cell.mass + gravity) * timeStep;
            }
        });
    });
}

void MPMStandard::gridToParticle(sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        auto particles_acc = d_particles.get_access<sycl::access::mode::read_write>(cgh);
        auto grid_acc = d_grid.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<1>(d_particles.get_count()), [=](sycl::id<1> idx) {
            auto& particle = particles_acc[idx];
            sycl::double3 velocityUpdate(0, 0, 0);
            sycl::double3 baseNode = sycl::double3(particle.position / gridSpacing);

            for (int i = 0; i < 8; ++i) {
                sycl::double3 node = baseNode + sycl::double3((i & 1), ((i & 2) >> 1), ((i & 4) >> 2));
                sycl::double3 nodePos = sycl::double3(node) * gridSpacing;
                sycl::double3 diff = (particle.position - nodePos) / gridSpacing;

                double weight = (1 - diff.x()) * (1 - diff.y()) * (1 - diff.z());

                size_t nodeIdx = node.x() + node.y() * gridResolution.x() + node.z() * gridResolution.x() * gridResolution.y();
                const auto& cell = grid_acc[nodeIdx];

                velocityUpdate += cell.velocity * weight;
            }

            particle.velocity = velocityUpdate;
            particle.position += particle.velocity * timeStep;
        });
    });
}

void MPMStandard::updateDeformationGradient(sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        auto particles_acc = d_particles.get_access<sycl::access::mode::read_write>(cgh);
        auto grid_acc = d_grid.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<1>(d_particles.get_count()), [=](sycl::id<1> idx) {
            auto& particle = particles_acc[idx];
            Matrix3d velocityGradient = Matrix3d::Zero();
            sycl::double3 baseNode = sycl::double3(particle.position / gridSpacing);

            for (int i = 0; i < 8; ++i) {
                sycl::double3 node = baseNode + sycl::double3((i & 1), ((i & 2) >> 1), ((i & 4) >> 2));
                sycl::double3 nodePos = sycl::double3(node) * gridSpacing;
                sycl::double3 diff = (particle.position - nodePos) / gridSpacing;

                sycl::double3 weight_grad(
                    (i & 1) ? 1 : -1,
                    ((i & 2) >> 1) ? 1 : -1,
                    ((i & 4) >> 2) ? 1 : -1
                );
                weight_grad /= gridSpacing;

                size_t nodeIdx = node.x() + node.y() * gridResolution.x() + node.z() * gridResolution.x() * gridResolution.y();
                const auto& cell = grid_acc[nodeIdx];

                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        velocityGradient(j, k) += cell.velocity[j] * weight_grad[k];
                    }
                }
            }

            Matrix3d deformationGradientUpdate = Matrix3d::Identity() + velocityGradient * timeStep;
            particle.deformationGradient = deformationGradientUpdate * particle.deformationGradient;
        });
    });
}

sycl::double3 MPMStandard::interpolate(const sycl::double3& position, sycl::queue& queue) {
    sycl::double3 result(0, 0, 0);
    sycl::double3 baseNode = sycl::double3(position / gridSpacing);

    queue.submit([&](sycl::handler& cgh) {
        auto grid_acc = d_grid.get_access<sycl::access::mode::read>(cgh);

        cgh.single_task([=, &result]() {
            for (int i = 0; i < 8; ++i) {
                sycl::double3 node = baseNode + sycl::double3((i & 1), ((i & 2) >> 1), ((i & 4) >> 2));
                sycl::double3 nodePos = sycl::double3(node) * gridSpacing;
                sycl::double3 diff = (position - nodePos) / gridSpacing;

                double weight = (1 - diff.x()) * (1 - diff.y()) * (1 - diff.z());

                size_t nodeIdx = node.x() + node.y() * gridResolution.x() + node.z() * gridResolution.x() * gridResolution.y();
                const auto& cell = grid_acc[nodeIdx];

                result += cell.velocity * weight;
            }
        });
    });
    queue.wait_and_throw();

    return result;
}

Matrix3d MPMStandard::computeStress(const Matrix3d& deformationGradient, const Particle& particle) {
    // Compute the Jacobian
    double J = deformationGradient.Determinant();
    J = std::max(J, 1e-10);  // Prevent division by zero

    // Compute the volumetric and deviatoric parts of the deformation gradient
    double J_1_3 = custom_cbrt(J);
    Matrix3d F_vol = Matrix3d::Identity() * J_1_3;
    Matrix3d F_dev = deformationGradient / J_1_3;

    // Compute the strain
    Matrix3d strain = 0.5 * (F_dev.Transpose() * F_dev - Matrix3d::Identity());

    // Compute the stress using a Neo-Hookean model
    double mu = shearModulus;
    double lambda = bulkModulus - 2.0/3.0 * shearModulus;

    double Trace_strain = strain.Trace();
    Matrix3d stress = 2.0 * mu * strain + lambda * Trace_strain * Matrix3d::Identity();
    stress += mu * (F_dev.Transpose() * F_dev - Matrix3d::Identity());

    // Add volumetric stress
    stress += bulkModulus * (J - 1) * Matrix3d::Identity();

    // Check for yield criterion (e.g., von Mises yield criterion)
    Matrix3d s = stress - (stress.Trace() / 3.0) * Matrix3d::Identity();
    double sum_of_squares = 0.0;
        for (int i = 0; i < 9; ++i) {
            sum_of_squares += s.Array()[i] * s.Array()[i];
        }
    double vonMises = std::sqrt(1.5 * sum_of_squares);
    
    // Apply plasticity model
    double scale = std::min(1.0, particle.yieldStress / std::max(vonMises, 1e-10));
    stress *= scale;

    // Apply damage model
    double damageEffect = 1.0 - particle.damageParameter;
    stress *= damageEffect;

    return stress;
}

} // namespace gm_mpm