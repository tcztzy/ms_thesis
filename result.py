#%%

import pathlib
import pickle
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

from main import run
from moo import MyProblem, OriginalProblem, iv, pick, sv

if pathlib.Path("result_demo.csv").exists():
    df = pd.read_csv("result_demo.csv", index_col=0, parse_dates=[1])
else:

    df = pd.DataFrame(
        columns=[
            "date",
            "light_interception",
            "lint_yield",
            "leaf_area_index",
            "seed_cotton_yield",
            "plant_height",
            "main_stem_nodes",
            "leaf_weight",
            "petiole_weight",
            "stem_weight",
            "number_of_squares",
            "number_of_green_bolls",
            "number_of_open_bolls",
            "square_weight",
            "boll_weight",
            "root_weight",
            "plant_weight",
            "swc0-10",
            "swc0-20",
            "swc0-30",
            "swc1-10",
            "swc1-20",
            "swc1-30",
            "swc2-10",
            "swc2-20",
            "swc2-30",
            "swc3-10",
            "swc3-20",
            "swc3-30",
            "lai00",
            "lai01",
            "lai02",
            "lai03",
            "lai04",
            "lai05",
            "lai06",
            "lai07",
            "lai08",
            "lai09",
            "lai10",
            "lai11",
            "lai12",
            "lai13",
            "lai14",
            "lai15",
            "lai16",
            "lai17",
            "lai18",
            "lai19",
            "treatment",
            "version",
        ]
    )

    for version in (0, 2):
        with open(f"moo_result_{version}.pkl", "rb") as f:
            res = pickle.load(f)
        i = pick(res.F)
        i = 9 if version == 0 else 25
        x = res.X[i]
        if version == 2:
            # x[3] = 3.458
            pass
        X = np.append(iv.copy(), np.ones(20) * -3)
        for i, v in zip(sv, x):
            X[i] = v
        if version != 2:
            X = X[:-20]
        for year in (2019, 2020):
            for i in range(6):
                d = run(X, year, i, version)
                d["treatment"] = f"T{i + 1}"
                d["version"] = "Original" if version == 0 else "Modified"
                df = df.append(d, ignore_index=True)
    df.to_csv("result_demo.csv")
li = pd.read_excel("LI_MEAN.xlsx", index_col=0)
bio = pd.read_excel("BIO_FINAL.xlsx", index_col=0)
bio["treatment"] = bio["treatment"].str.replace("W", "T")

lai = pd.read_excel("LAI_FINAL.xlsx", index_col=0)
lai["treatment"] = lai["treatment"].str.replace("W", "T")
lai_plot = (
    df[df["date"].isin(lai["date"])]
    .merge(lai, on=["date", "treatment"])
    .rename(
        columns={
            "LAI": "Measured LAI",
            "leaf_area_index": "Simulated LAI",
            "treatment": "Treatment",
            "version": "LI method",
        }
    )
)
seedcotton_yield = (
    pd.read_excel("seedcotton_yield.xlsx", index_col=[0, 1])
    .reset_index()
    .rename(
        columns={
            0: "Measured SCY (kg/ha)",
            "level_0": "Year",
            "level_1": "Treatment",
        }
    )
)
seedcotton_yield["Treatment"] = seedcotton_yield["Treatment"].str.replace("W", "T")
x = (
    df.query("date == '2019-10-25' or date == '2020-10-25'")
    .sort_values(["date", "treatment"])
    .rename(
        columns={
            "treatment": "Treatment",
            "seed_cotton_yield": "Simulated SCY (kg/ha)",
            "version": "LI method",
        }
    )
)
x["Year"] = (x["date"] > "2020-01-01").astype(int) + 2019
scy = x.merge(seedcotton_yield, on=["Year", "Treatment"], how="left")
fig, axes = plt.subplots(3, 2, figsize=(8, 12), dpi=800)
o = lai_plot["LI method"] == "Original"
ax = axes[0, 0]
ax.set_aspect("equal")
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 4.5)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "Original: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                lai_plot[o]["Measured LAI"], lai_plot[o]["Simulated LAI"]
            )
        )
        / lai_plot[o]["Measured LAI"].mean(),
        np.sqrt(
            mean_squared_error(
                lai_plot[~o]["Measured LAI"], lai_plot[~o]["Simulated LAI"]
            )
        )
        / lai_plot[~o]["Measured LAI"].mean(),
    ),
    transform=ax.transAxes,
)


def anno_pos(l, u):
    return 0.1 * (u - l) + l, 0.9 * (u - l) + l


ax.text(0.05, 0.95, "(a)", transform=ax.transAxes)
sns.scatterplot(
    x="Measured LAI",
    y="Simulated LAI",
    style="LI method",
    color="k",
    data=lai_plot,
    legend=False,
    ax=ax,
)

ax = axes[1, 1]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = scy["LI method"] == "Original"
ax.text(
    0.05,
    0.83,
    "(d)\nOriginal: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                scy[o]["Measured SCY (kg/ha)"], scy[o]["Simulated SCY (kg/ha)"]
            )
        )
        / scy[o]["Measured SCY (kg/ha)"].mean(),
        np.sqrt(
            mean_squared_error(
                scy[~o]["Measured SCY (kg/ha)"], scy[~o]["Simulated SCY (kg/ha)"]
            )
        )
        / scy[~o]["Measured SCY (kg/ha)"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(2500, 6500)
ax.set_ylim(2500, 6500)
sns.scatterplot(
    x="Measured SCY (kg/ha)",
    y="Simulated SCY (kg/ha)",
    style="LI method",
    color="k",
    legend=False,
    data=scy,
    ax=ax,
)

li_x = (
    li.stack()
    .reset_index()
    .rename(columns={"level_0": "date", "level_1": "Treatment", 0: "Measured LI"})
)
li_x["Treatment"] = li_x["Treatment"].str.replace("W", "T")
li_x = li_x.merge(
    df.rename(
        columns={
            "treatment": "Treatment",
            "version": "LI method",
            "light_interception": "Simulated LI",
        }
    ),
    on=["date", "Treatment"],
)
ax = axes[0, 1]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = li_x["LI method"] == "Original"

ax.text(
    0.05,
    0.84,
    "(b)\nOriginal: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(mean_squared_error(li_x[o]["Measured LI"], li_x[o]["Simulated LI"]))
        / li_x[o]["Measured LI"].mean(),
        np.sqrt(mean_squared_error(li_x[~o]["Measured LI"], li_x[~o]["Simulated LI"]))
        / li_x[~o]["Measured LI"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(0.3, 1)
ax.set_ylim(0.3, 1)
sns.scatterplot(
    x="Measured LI",
    y="Simulated LI",
    style="LI method",
    color="k",
    legend=False,
    data=li_x,
    ax=ax,
)

height = bio.merge(df, on=["date", "treatment"]).rename(
    columns={
        "version": "LI method",
        "plant_height": "Simulated plant height (cm)",
        "Plant height (cm)": "Measured plant height (cm)",
    }
)
ax = axes[2, 0]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = height["LI method"] == "Original"
ax.text(0.05, 0.95, "(e)", transform=ax.transAxes)
ax.text(
    0.55,
    0.05,
    "Original: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                height[o]["Measured plant height (cm)"],
                height[o]["Simulated plant height (cm)"],
            )
        )
        / height[o]["Measured plant height (cm)"].mean(),
        np.sqrt(
            mean_squared_error(
                height[~o]["Measured plant height (cm)"],
                height[~o]["Simulated plant height (cm)"],
            )
        )
        / height[~o]["Measured plant height (cm)"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(20, 110)
ax.set_ylim(20, 110)
sns.scatterplot(
    x="Measured plant height (cm)",
    y="Simulated plant height (cm)",
    style="LI method",
    color="k",
    legend=False,
    data=height,
    ax=ax,
)

ax = axes[1, 0]

ax.text(0.05, 0.95, "(c)", transform=ax.transAxes)
tagb = bio.merge(df, on=["date", "treatment"]).rename(
    columns={
        "version": "LI method",
        "plant_weight": "Simulated TAGB (kg/ha)",
        "Above ground biomass (kg/ha)": "Measured TAGB (kg/ha)",
    }
)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "Original: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                tagb[o]["Measured TAGB (kg/ha)"],
                tagb[o]["Simulated TAGB (kg/ha)"],
            )
        )
        / tagb[o]["Measured TAGB (kg/ha)"].mean(),
        np.sqrt(
            mean_squared_error(
                tagb[~o]["Measured TAGB (kg/ha)"],
                tagb[~o]["Simulated TAGB (kg/ha)"],
            )
        )
        / tagb[~o]["Measured TAGB (kg/ha)"].mean(),
    ),
    transform=ax.transAxes,
)
sns.scatterplot(
    x="Measured TAGB (kg/ha)",
    y="Simulated TAGB (kg/ha)",
    style="LI method",
    color="k",
    legend=False,
    data=tagb,
    ax=ax,
)
ax = axes[2, 1]

ax.text(0.05, 0.95, "(f)", transform=ax.transAxes)
nodes = (
    bio.merge(df, on=["date", "treatment"])
    .fillna(0)
    .rename(
        columns={
            "version": "LI method",
            "main_stem_nodes": "Simulated mainstem nodes",
            "Mainstem nodes": "Measured mainstem nodes",
        }
    )
)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "Original: {:.2%}\nModified: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                nodes[o]["Measured mainstem nodes"],
                nodes[o]["Simulated mainstem nodes"],
            )
        )
        / nodes[o]["Measured mainstem nodes"].mean(),
        np.sqrt(
            mean_squared_error(
                nodes[~o]["Measured mainstem nodes"],
                nodes[~o]["Simulated mainstem nodes"],
            )
        )
        / nodes[~o]["Measured mainstem nodes"].mean(),
    ),
    transform=ax.transAxes,
)
sns.scatterplot(
    x="Measured mainstem nodes",
    y="Simulated mainstem nodes",
    style="LI method",
    color="k",
    data=nodes,
    ax=ax,
)
plt.legend(bbox_to_anchor=(-0.3, -0.17), loc="upper left", borderaxespad=0)
plt.subplots_adjust(hspace=0.22)
plt.savefig("1v1.png")