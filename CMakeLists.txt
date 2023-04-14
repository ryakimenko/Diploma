cmake_minimum_required(VERSION 3.23)
project(Diploma)

set(BLA_VENDOR OpenBLAS)

find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)
set(OpenBLAS_LIBNAME openblas${SUFFIX64_UNDERSCORE})

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-march=znver3 -fvectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -ffast-math -O3)
add_compile_options(-gdwarf-4)

add_executable(Diploma main.cpp)

if (BLAS_FOUND)
    target_link_libraries(Diploma ${BLAS_LIBRARIES})
else()
    message("Blas not found")
endif()