#!/usr/bin/env python3
# Author: Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_lc(
    bts_table, noisy_table, sig_noise_mask: bool = True, fig_size: tuple = (8, 5)
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    peakjd = float(bts_table.meta["bts_peak_jd"])

    if sig_noise_mask:
        s_n = np.abs(np.array(noisy_table["flux"] / noisy_table["fluxerr"]))
        mask = s_n > 3.0
        noisy_table = noisy_table[mask]

    col = []
    col_noisy = []

    for i in range(len(bts_table)):
        if bts_table["band"][i] == "ztfr":
            col.append("crimson")
        elif bts_table["band"][i] == "ztfg":
            col.append("forestgreen")
        elif bts_table["band"][i] == "ztfi":
            col.append("darkorange")
        else:
            col.append("k")

    for i in range(len(noisy_table)):
        if noisy_table["band"][i] == "ztfr":
            col_noisy.append("orchid")
        elif noisy_table["band"][i] == "ztfg":
            col_noisy.append("mediumblue")
        elif noisy_table["band"][i] == "ztfi":
            col_noisy.append("goldenrod")
        else:
            col_noisy.append("k")

    ax.errorbar(
        bts_table["jd"] - peakjd,
        bts_table["magpsf"],
        yerr=bts_table["sigmapsf"],
        ecolor=col,
        ls="none",
        elinewidth=1.7,
    )
    ax.scatter(bts_table["jd"] - peakjd, bts_table["magpsf"], c=col, marker="o", s=20)

    ax.errorbar(
        noisy_table["jd"] - peakjd,
        noisy_table["magpsf"],
        yerr=noisy_table["sigmapsf"],
        ecolor=col_noisy,
        ls="none",
        elinewidth=1.7,
    )
    ax.scatter(
        noisy_table["jd"] - peakjd, noisy_table["magpsf"], c=col_noisy, marker="o", s=20
    )

    # legend maker
    ax.scatter(
        1000,
        0,
        c="crimson",
        s=20,
        label=r"r band at $z = %.2f$" % (float(bts_table.meta["bts_z"])),
    )
    ax.scatter(
        1000,
        0,
        c="forestgreen",
        s=20,
        label=r"g band at $z = %.2f$" % (float(bts_table.meta["bts_z"])),
    )
    ax.scatter(
        1000,
        0,
        c="darkorange",
        s=20,
        label=r"i band lc at $z = %.2f$" % (float(bts_table.meta["bts_z"])),
    )
    ax.scatter(
        1000,
        0,
        c="orchid",
        s=20,
        label=r"r band at $z = %.2f$" % (float(noisy_table.meta["z"])),
    )
    ax.scatter(
        1000,
        0,
        c="mediumblue",
        s=20,
        label=r"g band at $z = %.2f$" % (float(noisy_table.meta["z"])),
    )
    ax.scatter(
        1000,
        0,
        c="goldenrod",
        s=20,
        label=r"i band lc at $z = %.2f$" % (float(noisy_table.meta["z"])),
    )

    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time after peak (days)")
    ax.set_ylim(np.nanmax(noisy_table["magpsf"]) + 0.3, min(bts_table["magpsf"]) - 0.3)
    ax.set_xlim(-10, 35)
    ax.legend()

    return ax
