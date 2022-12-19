/*
8 - Mandelbrot set, visualised
*/

#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string>
#include "draw_bmp.hpp"

using namespace std;

#define MAX_ITERATIONS 255
#define BATCH_SIZE 50

double GetTime() {
   struct timeval time;
   gettimeofday(&time, NULL);
   double res = (double)time.tv_sec+(double)time.tv_usec/1000000;
   return res;
}

uint8_t is_mandelbrot(long double r0, long double i0) {
    long double r = 0;
    long double i = 0;
    uint32_t iterations = 0;
    long double r_temp;

    for (;(r*r + i*i <= 4) && (iterations < MAX_ITERATIONS); ++iterations) {
        r_temp = r*r - i*i + r0;
        i = 2*r*i + i0;
        r = r_temp;
    }

    return iterations;
}

long double pixel_to_real(long double dimension, uint32_t p) {
    return ((long double)2.5 / dimension * p -2);
}

long double pixel_to_imaginary(long double dimension, uint32_t p) {
    return (long double)2.5 / dimension * p -1.25;
}

void calc(BMP &bmp, uint32_t dimension) {
    uint8_t iterations = 0;
    int core_count, id;
    MPI_Comm_size(MPI_COMM_WORLD, &core_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (uint32_t i = id*BATCH_SIZE; i < dimension; i++) { // imaginary, vertical
        for (uint32_t r = 0; r < dimension; ++r) { // real, horizontal
            long double ic = pixel_to_imaginary((long double)dimension, i);
            long double rc = pixel_to_real((long double)dimension, r);
            iterations = is_mandelbrot(rc, ic);
            if (iterations == MAX_ITERATIONS) {
                bmp.set_pixel(r, i, 0, 0, 0);
            } else {
                bmp.set_pixel(r, i, 255, iterations, 0);
            }
        }

        if ((i+1) % BATCH_SIZE == 0) {
            i += (core_count - 1) * BATCH_SIZE;
        }
    }
}

void master(BMP &bmp, uint32_t dimension) {
    int core_count;
    MPI_Comm_size(MPI_COMM_WORLD, &core_count);

    for (int i = 0; i < dimension; i += BATCH_SIZE) {
        if (i % (core_count * BATCH_SIZE) == 0) continue;
        // cout << "Recv: " << i << endl;
        MPI_Recv(bmp.get_row_address(i), dimension * 3 * BATCH_SIZE, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void slave(BMP &bmp, uint32_t dimension) {
    MPI_Request request = MPI_REQUEST_NULL;
    int core_count, id;
    MPI_Comm_size(MPI_COMM_WORLD, &core_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (uint32_t i = id*BATCH_SIZE; i < dimension; i += core_count * BATCH_SIZE) { // imaginary, vertical
        // cout << id << " send: " << i << endl;
        MPI_Isend(bmp.get_row_address(i), dimension * 3 * BATCH_SIZE, MPI_UNSIGNED_CHAR, 0, i, MPI_COMM_WORLD, &request);
    }
}

int main(int argc, char *argv[]) {
    uint32_t dimension = 3000;
    int id;
    BMP bmp(dimension, dimension, false);

    // Parallelisable
    MPI_Init(&argc, &argv);
    double start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    calc(bmp, dimension);
    if (id == 0) {
        master(bmp, dimension);
    } else {
        slave(bmp, dimension);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();
    MPI_Finalize(); 

    // Non parallelisable
    if (id == 0) {
        // bmp.write("mandelbrot.bmp");
        cout << "time: " << end - start << endl;
    }

    return 0;
}