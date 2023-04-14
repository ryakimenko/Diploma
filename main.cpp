#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include "/opt/OpenBLAS/include/cblas.h"

constexpr int SM_d = 64 / sizeof(double);
constexpr int SM_f = 64 / sizeof(float);


void dgemm_block(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, int N) {
    long long int i, i2, j, j2, k, k2;

    for (i = 0; i < N; i += SM_d) {
        for (j = 0; j < N; j += SM_d) {
            for (k = 0; k < N; k += SM_d) {
                for (i2 = 0; i2 < SM_d; ++i2) {
                    for (k2 = 0; k2 < SM_d; ++k2) {
                        for (j2 = 0; j2 < SM_d; ++j2) {
                            C[i * N + j + j2] += A[i * N + k + k2] * B[k * N + j + j2];
                        }
                    }
                }
            }
        }
    }
}

void sgemm_block(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int N) {
    long long int i, i2, j, j2, k, k2;

    for (i = 0; i < N; i += SM_f) {
        for (j = 0; j < N; j += SM_f) {
            for (k = 0; k < N; k += SM_f) {
                for (i2 = 0; i2 < SM_f; ++i2) {
                    for (k2 = 0; k2 < SM_f; ++k2) {
                        for (j2 = 0; j2 < SM_f; ++j2) {
                            C[i * N + j + j2] += A[i * N + k + k2] * B[k * N + j + j2];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {

    long long N = atoi(argv[1]);
    std::string method = argv[2];
    std::ofstream outFile(argv[3]);
    std::string type = argv[4];


    double *__restrict__ C_d;
    double *__restrict__ A_d;
    double *__restrict__ B_d;

    float *__restrict__ C_f;
    float *__restrict__ A_f;
    float *__restrict__ B_f;

    int step = 100;


    std::chrono::time_point<std::chrono::steady_clock> start;

    if (type == "double") {
        double performance = 0.0;
        for (long long  matrixSize = 100; matrixSize <= N; matrixSize += step) {
            C_d = new double[matrixSize * matrixSize];
            A_d = new double[matrixSize * matrixSize];
            B_d = new double[matrixSize * matrixSize];

            for (int i = 0; i < matrixSize; i++) {
                for (int j = 0; j < matrixSize; j++) {
                    A_d[i * matrixSize + j] = (i + 1.0) * (j + 1.0) * 1.0;
                    B_d[i * matrixSize + j] = (i + 1.0) * (j + 1.0) * 1.0;
                    C_d[i * matrixSize + j] = 0.0;
                }
            }
            for(int repeats = 0; repeats < 5; repeats++) {
                start = std::chrono::steady_clock::now();
                if (method == "MoA") {
                    dgemm_block(A_d, B_d, C_d, matrixSize);
                }
                if (method == "OpenBLAS") {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, matrixSize, matrixSize, matrixSize, 1.0, A_d,
                                matrixSize, B_d, matrixSize, 0.0, C_d, matrixSize);
                }
                auto end = std::chrono::steady_clock::now();
                auto sec =  std::chrono::duration<double>(end - start).count();
                performance +=(2 * (matrixSize * matrixSize * matrixSize)) / sec / 1.0e9;
            }
            performance /= 5;
            outFile << matrixSize << ' ' << performance << '\n';
            delete[] A_d;
            delete[] B_d;
            delete[] C_d;
        }
    } else if (type == "float") {
        double performance = 0.0;
        for (long long matrixSize = 100; matrixSize <= N; matrixSize += step) {
            C_f = new float[matrixSize * matrixSize];
            A_f = new float[matrixSize * matrixSize];
            B_f = new float[matrixSize * matrixSize];

            for (int i = 0; i < matrixSize; i++) {
                for (int j = 0; j < matrixSize; j++) {
                    A_f[i * matrixSize + j] = (i + 1.0) * (j + 1.0) * 1.0;
                    B_f[i * matrixSize + j] = (i + 1.0) * (j + 1.0) * 1.0;
                    C_f[i * matrixSize + j] = 0.0;
                }
            }
            for(int repeats = 0; repeats < 5; repeats++) {
                start = std::chrono::steady_clock::now();
                if (method == "MoA") {
                    sgemm_block(A_f, B_f, C_f, matrixSize);
                }
                if (method == "OpenBLAS") {
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, matrixSize, matrixSize, matrixSize, 1.0, A_f,
                                matrixSize, B_f, N, 0.0, C_f, matrixSize);
                }
                auto end = std::chrono::steady_clock::now();
                auto sec =  std::chrono::duration<double>(end - start).count();
                performance +=(2 * (matrixSize * matrixSize * matrixSize)) / sec / 1.0e9;
            }

            performance /= 5;
            outFile << matrixSize << ' ' << performance << '\n';
            delete[] A_f;
            delete[] B_f;
            delete[] C_f;
        }
    }

    outFile.close();

    return 0;
}
