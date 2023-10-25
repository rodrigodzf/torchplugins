#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <fftw3.h>

#define SAMPLES 64
using FixedPointsArray = std::array<double, SAMPLES>;

class ShapeFFT
{
public:
    ShapeFFT()
    {
        // Initialize the FFT
        cfg = fftw_plan_dft_1d(
            SAMPLES,
            src.data(),
            dst.data(),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    void fft_magnitude(
        const FixedPointsArray &x,
        const FixedPointsArray &y
    )
    {
        // Copy input to src
        for(int i = 0; i < SAMPLES; i++)
        {
            src[i][0] = (float)x[i];
            src[i][1] = (float)y[i];
        } 

        // Perform the FFT
        fftw_execute(cfg);

        // Calculate the magnitudes
        for(int i = 0; i < SAMPLES; i++)
        {
            // Normalised magnitude
            mag[i] = sqrtf(dst[i][0] * dst[i][0] + dst[i][1] * dst[i][1]) / SAMPLES; 
        }
    }


public:
    FixedPointsArray mag;     // A magnitude array for the transformed data

private:
    // allocate the memory for the FFT in the heap (32 complex numbers)
    
    std::array<fftw_complex, SAMPLES> src; // A source array of input data
    std::array<fftw_complex, SAMPLES> dst; // A destination array for the transformed data
    fftw_plan cfg;
};
