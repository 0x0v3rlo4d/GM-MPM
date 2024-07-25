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

    // Zero matrix
    static Matrix3d Zero() {
        return Matrix3d();
    }

    // Identity matrix
    static Matrix3d Identity() {
        Matrix3d m;
        m(0,0) = m(1,1) = m(2,2) = 1.0;
        return m;
    }

    // Matrix multiplication
    Matrix3d operator*(const Matrix3d& other) const {
        Matrix3d result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i,j) = 0;
                for (int k = 0; k < 3; ++k) {
                    result(i,j) += (*this)(i,k) * other(k,j);
                }
            }
        }
        return result;
    }

    // Scalar multiplication
    Matrix3d operator*(double scalar) const {
        Matrix3d result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }

    // Scalar division
    Matrix3d operator/(double scalar) const {
        Matrix3d result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = this->data[i] / scalar;
        }
        return result;
    }

    // Addition
    Matrix3d operator+(const Matrix3d& other) const {
        Matrix3d result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    // Subtraction
    Matrix3d operator-(const Matrix3d& other) const {
        Matrix3d result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    // Addition assignment
    Matrix3d& operator+=(const Matrix3d& other) {
        for (int i = 0; i < 9; ++i) {
            this->data[i] += other.data[i];
        }
        return *this;
    }

    // Multiplication assignment
    Matrix3d& operator*=(double scalar) {
        for (int i = 0; i < 9; ++i) {
            this->data[i] *= scalar;
        }
        return *this;
    }

    // Determinant
    double Determinant() const {
        return data[0] * (data[4] * data[8] - data[5] * data[7]) -
               data[1] * (data[3] * data[8] - data[5] * data[6]) +
               data[2] * (data[3] * data[7] - data[4] * data[6]);
    }

    // Transpose
    Matrix3d Transpose() const {
        Matrix3d result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    // Trace
    double Trace() const {
        return data[0] + data[4] + data[8];
    }

    // Array (returns a reference to the underlying array)
    double* Array() {
        return data;
    }

    const double* Array() const {
        return data;
    }
};

// Non-member scalar multiplication
inline Matrix3d operator*(double scalar, const Matrix3d& matrix) {
    return matrix * scalar;
}

struct Particle {
    sycl::double3 position;
    sycl::double3 velocity;
    Matrix3d deformationGradient;
    double mass;
    double volume;
    
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