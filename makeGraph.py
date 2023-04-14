#!/usr/bin/env python3
# encoding=utf8

import matplotlib.pyplot as plt
import numpy as np


def main():
    datafiles = ["cmake-build-debug/MoAdouble.dat","cmake-build-debug/BLASdouble.dat"]

    cm = 1/2.54  # centimeters in inche
    fig = plt.figure(figsize=(32*cm, 20*cm))
    ax = fig.add_subplot(111)
    colors=["red", "green"]
    ax.set_title("Clang-16, datatype - double")
    ax.set(xlabel="Matrix Dimension N", ylabel="GigaFlops/sec")
    plt.grid("true")
    plt.yscale("log")
    plt.xlim([0,2500])
    label=["dgemm_block","OpenBLAS dgemm"]

    # Draw data files
    for i in range(len(datafiles)):

        data = np.loadtxt(datafiles[i])
        x = data[:, 0]
        y = data[:, 1]
        #plt.scatter(x,y)
        ax.plot(x,y,'o-',markersize=2,c=colors[i], label = label[i])
    plt.legend()

    fig.savefig('Graph.png', dpi=300)

if __name__ == "__main__":
    main()
