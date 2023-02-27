#!/usr/bin/env python3
# Author: Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os

import matplotlib as mpl  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd  # type:ignore
import seaborn as sns  # type:ignore

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
    phase_limit: bool = True,
    sig_noise_mask: bool = True,
    fig_size: tuple = (8, 5),
    plot_iband=False,
    output_format: str = "parsnip",
):
    assert output_format in ["parsnip", "ztfnuclear"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    peakjd = float(bts_table.meta["bts_peak_jd"])

    if not plot_iband:
        bts_table = bts_table[bts_table["band"] != "ztfi"]
        noisy_table = noisy_table[noisy_table["band"] != "ztfi"]

    if output_format == "parsnip":
        flux_col = "flux"
        flux_err_col = "fluxerr"
        date_col = "jd"
        peak = peakjd
    elif output_format == "ztfnuclear":
        flux_col = "ampl_corr"
        flux_err_col = "ampl_err_corr"
        date_col = "obsmjd"
        peak = peakjd - 2400000.5

    if sig_noise_mask:
        s_n_noisy = np.abs(np.array(noisy_table[flux_col] / noisy_table[flux_err_col]))
        s_n_orig = np.abs(np.array(bts_table[flux_col] / bts_table[flux_err_col]))
        mask_noisy = s_n_noisy > 5.0
        mask_orig = s_n_orig > 5.0
        noisy_table = noisy_table[mask_noisy]
        bts_table = bts_table[mask_orig]

    if len(noisy_table) == 0:
        return None

    bts = bts_table.to_pandas()
    bts["type"] = "orig"
    noisy = noisy_table.to_pandas()
    noisy["type"] = "noisy"

    config = {
        "orig": {
            "colors": {"ztfg": "forestgreen", "ztfr": "crimson", "ztfi": "darkorange"},
            "header": bts_table.meta,
            "z": float(bts_table.meta["bts_z"]),
        },
        "noisy": {
            "colors": {"ztfg": "mediumblue", "ztfr": "orchid", "ztfi": "goldenrod"},
            "header": noisy_table.meta,
            "z": float(noisy_table.meta["z"]),
        },
    }
    headers = {"orig": bts_table.meta, "noisy": noisy_table.meta}

    if plot_iband:
        bands_to_use = ["ztfg", "ztfr", "ztfi"]
    else:
        bands_to_use = ["ztfg", "ztfr"]

    df = pd.concat([bts, noisy])
    for band in bands_to_use:
        for lc_type in ["orig", "noisy"]:
            label = f"${band[-1:]}$-band at $z={config[lc_type]['z']:.2f}$"
            _df = df.query("type==@lc_type and band==@band")
            ax.errorbar(
                x=_df[date_col] - peak,
                y=_df.magpsf,
                yerr=_df.sigmapsf,
                ecolor=config[lc_type]["colors"][band],
                ls="none",
                fmt=".",
                c=config[lc_type]["colors"][band],
                elinewidth=1.2,
                label=label,
            )

    ax.set_ylabel("Magnitude (AB)")
    ax.set_xlabel("Time after peak (days)")
    ax.set_ylim(np.nanmax(noisy_table["magpsf"]) + 0.3, min(bts_table["magpsf"]) - 0.3)
    ax.set_title(str(bts_table.meta["name"] + " : " + str(bts_table.meta["bts_class"])))

    if phase_limit:
        ax.set_xlim(-10, 35)

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
                all_old_mag.extend(lc["magpsf"])
                all_old_mag_err.extend(lc["sigmapsf"])
            elif key == "bts_noisified":
                all_new_mag.extend(lc["magpsf"])
                all_new_mag_err.extend(lc["sigmapsf"])

    # Get data in right format for plotting
    all_old_mag = np.array(all_old_mag)
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
    df2 = df[
        (df["Magnitude"] < 22.0)
        & (df["Magnitude error"] < 1.0)
        & (df["Magnitude"] > 17.0)
    ]

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
        hue_order=["Simulated", "Real"],
        kind="kde",
        fill=True,
        alpha=0.7,
    )
    colours = ["darkorange"] * 8 + ["steelblue"] * 8
    colours = np.array(colours)
    g.ax_joint.errorbar(
        df_scatter["Bins"][df_scatter["Data type"] == "Real"],
        df_scatter["Mean error"][df_scatter["Data type"] == "Real"],
        yerr=df_scatter["Std error"][df_scatter["Data type"] == "Real"],
        c="darkorange",
        ls="none",
        capsize=3.0,
    )
    g.ax_joint.errorbar(
        df_scatter["Bins"][df_scatter["Data type"] == "Simulated"],
        df_scatter["Mean error"][df_scatter["Data type"] == "Simulated"],
        yerr=df_scatter["Std error"][df_scatter["Data type"] == "Simulated"],
        c="steelblue",
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
