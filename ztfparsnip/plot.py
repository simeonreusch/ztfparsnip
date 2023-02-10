#!/usr/bin/env python3
# Author: Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

plt.rc("font", family="serif", size=10)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
mpl.rcParams["ytick.major.size"] = 5
mpl.rcParams["ytick.major.width"] = 1
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["ytick.minor.width"] = 1
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["xtick.major.width"] = 1


def plot_lc(
    bts_table,
    noisy_table,
    sig_noise_mask: bool = True,
    fig_size: tuple = (8, 5),
    plot_iband=False,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    peakjd = float(bts_table.meta["bts_peak_jd"])

    if not plot_iband:
        bts_table = bts_table[bts_table["band"] != "ztfi"]
        noisy_table = noisy_table[noisy_table["band"] != "ztfi"]

    if sig_noise_mask:
        s_n = np.abs(np.array(noisy_table["flux"] / noisy_table["fluxerr"]))
        mask = s_n > 3.0
        noisy_table = noisy_table[mask]

    if len(noisy_table) == 0:
        return None

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
        elinewidth=1.2,
    )
    ax.scatter(bts_table["jd"] - peakjd, bts_table["magpsf"], c=col, marker="o", s=20)

    ax.errorbar(
        noisy_table["jd"] - peakjd,
        noisy_table["magpsf"],
        yerr=noisy_table["sigmapsf"],
        ecolor=col_noisy,
        ls="none",
        elinewidth=1.2,
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
    if plot_iband:
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
    if plot_iband:
        ax.scatter(
            1000,
            0,
            c="goldenrod",
            s=20,
            label=r"i band lc at $z = %.2f$" % (float(noisy_table.meta["z"])),
        )

    ax.set_ylabel("Magnitude (AB)")
    ax.set_xlabel("Time after peak (days)")
    ax.set_ylim(np.nanmax(noisy_table["magpsf"]) + 0.3, min(bts_table["magpsf"]) - 0.3)
    # ax.set_xlim(-10, 35)
    ax.legend()

    return ax


def plot_magnitude_dist(lightcurve_dict):

    all_old_mag = []
    all_old_mag_err = []
    all_new_mag = []
    all_new_mag_err = []

    for key, lc_list in lightcurve_dict.items():
        for lc in lc_list:
            if key == "bts_orig":
                all_old_mag.append(lc["magpsf"])
                all_old_mag_err.append(lc["sigmapsf"])
            elif key == "bts_noisified":
                all_new_mag.append(lc["magpsf"])
                all_new_mag_err.append(lc["sigmapsf"])

    # Get data in right format for plotting
    all_old_mag = np.asarray(all_old_mag)
    all_old_mag_err = np.array(all_old_mag_err)
    all_new_mag = np.array(all_new_mag)
    all_new_mag_err = np.array(all_new_mag_err)
    mask = np.isfinite(all_new_mag) & np.isfinite(all_new_mag_err)
    all_new_mag = all_new_mag[mask]
    all_new_mag_err = all_new_mag_err[mask]

    real_arr = ["Real"] * len(all_old_mag)
    sim_arr = ["Simulated"] * len(all_new_mag)
    real_arr = np.array(real_arr)
    sim_arr = np.array(sim_arr)
    all_type = np.concatenate([real_arr, sim_arr])

    all_m = np.concatenate([all_old_mag, all_new_mag])
    all_e = np.concatenate([all_old_mag_err, all_new_mag_err])

    dat = {"Magnitude": all_m, "Magnitude error": all_e, "Data type": all_type}
    df = pd.DataFrame(data=dat)

    # Make cuts
    # df2 = df[(df['Magnitude'] < 22.) & (df['Magnitude error'] < 1.) & (df['Magnitude'] > 17.)]
    df2 = df.copy()

    # Make binned scatter points
    # sort on mag
    df2 = df2.sort_values(by="Magnitude", ignore_index=True)
    # create bins
    df2["Bin"] = pd.cut(df2["Magnitude"], 8, include_lowest=True)
    # group on bin
    group = df2.groupby("Bin")
    # list comprehension to split groups into list of dataframes
    dfs = [group.get_group(x) for x in group.groups]

    bins = []
    mean_err_real = []
    std_err_real = []
    mean_err_sim = []
    std_err_sim = []

    for df in dfs:
        df = df.reset_index(drop=True)
        bins.append(df["Bin"][0].mid)
        mean_err_real.append(df["Magnitude error"][df["Data type"] == "Real"].mean())
        std_err_real.append(df["Magnitude error"][df["Data type"] == "Real"].std())
        mean_err_sim.append(
            df["Magnitude error"][df["Data type"] == "Simulated"].mean()
        )
        std_err_sim.append(df["Magnitude error"][df["Data type"] == "Simulated"].std())

    bins = np.array(bins)
    mean_err_real = np.array(mean_err_real)
    std_err_real = np.array(std_err_real)
    mean_err_sim = np.array(mean_err_sim)
    std_err_sim = np.array(std_err_sim)

    real_arr_mean = ["Real"] * len(mean_err_real)
    sim_arr_mean = ["Simulated"] * len(mean_err_sim)
    real_arr_mean = np.array(real_arr_mean)
    sim_arr_mean = np.array(sim_arr_mean)
    all_type_mean = np.concatenate([real_arr_mean, sim_arr_mean])
    mean_err_all = np.concatenate([mean_err_real, mean_err_sim])
    std_err_all = np.concatenate([std_err_real, std_err_sim])
    bins_all = np.concatenate([bins, bins])

    datat = {
        "Bins": bins_all,
        "Mean error": mean_err_all,
        "Std error": std_err_all,
        "Data type": all_type_mean,
    }
    df_scatter = pd.DataFrame(data=datat)

    # PLOT and save
    g = sns.jointplot(
        data=df2,
        x="Magnitude",
        y="Magnitude error",
        hue="Data type",
        kind="kde",
        fill=True,
        alpha=0.6,
    )
    colours = ["steelblue"] * 8 + ["darkorange"] * 8
    colours = np.array(colours)
    g.ax_joint.errorbar(
        df_scatter["Bins"][df_scatter["Data type"] == "Real"],
        df_scatter["Mean error"][df_scatter["Data type"] == "Real"],
        yerr=df_scatter["Std error"][df_scatter["Data type"] == "Real"],
        c="steelblue",
        ls="none",
        capsize=3.0,
    )
    g.ax_joint.errorbar(
        df_scatter["Bins"][df_scatter["Data type"] == "Simulated"],
        df_scatter["Mean error"][df_scatter["Data type"] == "Simulated"],
        yerr=df_scatter["Std error"][df_scatter["Data type"] == "Simulated"],
        c="darkorange",
        ls="none",
        capsize=3.0,
    )
    g.ax_joint.scatter(
        df_scatter["Bins"],
        df_scatter["Mean error"],
        c=colours,
        marker="o",
        s=20,
        edgecolors="k",
        linewidths=0.3,
    )
    return g
