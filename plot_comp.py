#%%

import pathlib
import pickle
from operator import attrgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

from main import run
from moo import MyProblem, OriginalProblem, iv, pick, sv

mpl.rc("font", family="SimSun")
mpl.rc("axes", unicode_minus=False)

if pathlib.Path("result_demo.csv").exists():
    df = pd.read_csv("result_demo.csv", index_col=0, parse_dates=[1])
    df["version"] = (
        df["version"].str.replace("Modified", "改版").replace("Original", "原版")
    )
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
                d["version"] = "原版" if version == 0 else "改版"
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
            "LAI": "实测叶面积指数",
            "leaf_area_index": "模拟叶面积指数",
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
            0: "实测籽棉产量 (kg/ha)",
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
            "seed_cotton_yield": "模拟籽棉产量 (kg/ha)",
            "version": "LI method",
        }
    )
)
x["Year"] = (x["date"] > "2020-01-01").astype(int) + 2019
scy = x.merge(seedcotton_yield, on=["Year", "Treatment"], how="left")
fig, axes = plt.subplots(3, 2, figsize=(8, 12), dpi=800)
o = lai_plot["LI method"] == "原版"
ax = axes[0, 0]
ax.set_aspect("equal")
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 4.5)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(mean_squared_error(lai_plot[o]["实测叶面积指数"], lai_plot[o]["模拟叶面积指数"]))
        / lai_plot[o]["实测叶面积指数"].mean(),
        np.sqrt(mean_squared_error(lai_plot[~o]["实测叶面积指数"], lai_plot[~o]["模拟叶面积指数"]))
        / lai_plot[~o]["实测叶面积指数"].mean(),
    ),
    transform=ax.transAxes,
)


def anno_pos(l, u):
    return 0.1 * (u - l) + l, 0.9 * (u - l) + l


ax.text(0.05, 0.95, "(a)", transform=ax.transAxes)
sns.scatterplot(
    x="实测叶面积指数",
    y="模拟叶面积指数",
    style="LI method",
    color="k",
    data=lai_plot,
    legend=False,
    ax=ax,
)

ax = axes[1, 1]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = scy["LI method"] == "原版"
ax.text(
    0.05,
    0.83,
    "(d)\n原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(mean_squared_error(scy[o]["实测籽棉产量 (kg/ha)"], scy[o]["模拟籽棉产量 (kg/ha)"]))
        / scy[o]["实测籽棉产量 (kg/ha)"].mean(),
        np.sqrt(
            mean_squared_error(scy[~o]["实测籽棉产量 (kg/ha)"], scy[~o]["模拟籽棉产量 (kg/ha)"])
        )
        / scy[~o]["实测籽棉产量 (kg/ha)"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(2500, 6500)
ax.set_ylim(2500, 6500)
sns.scatterplot(
    x="实测籽棉产量 (kg/ha)",
    y="模拟籽棉产量 (kg/ha)",
    style="LI method",
    color="k",
    legend=False,
    data=scy,
    ax=ax,
)

li_x = (
    li.stack()
    .reset_index()
    .rename(columns={"level_0": "date", "level_1": "Treatment", 0: "实测光能截获率"})
)
li_x["Treatment"] = li_x["Treatment"].str.replace("W", "T")
li_x = li_x.merge(
    df.rename(
        columns={
            "treatment": "Treatment",
            "version": "LI method",
            "light_interception": "模拟光能截获率",
        }
    ),
    on=["date", "Treatment"],
)
ax = axes[0, 1]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = li_x["LI method"] == "原版"

ax.text(
    0.05,
    0.84,
    "(b)\n原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(mean_squared_error(li_x[o]["实测光能截获率"], li_x[o]["模拟光能截获率"]))
        / li_x[o]["实测光能截获率"].mean(),
        np.sqrt(mean_squared_error(li_x[~o]["实测光能截获率"], li_x[~o]["模拟光能截获率"]))
        / li_x[~o]["实测光能截获率"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(0.3, 1)
ax.set_ylim(0.3, 1)
sns.scatterplot(
    x="实测光能截获率",
    y="模拟光能截获率",
    style="LI method",
    color="k",
    legend=False,
    data=li_x,
    ax=ax,
)

height = bio.merge(df, on=["date", "treatment"]).rename(
    columns={
        "version": "LI method",
        "plant_height": "模拟株高 (cm)",
        "Plant height (cm)": "实测株高 (cm)",
    }
)
ax = axes[2, 0]

ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
o = height["LI method"] == "原版"
ax.text(0.05, 0.95, "(e)", transform=ax.transAxes)
ax.text(
    0.55,
    0.05,
    "原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                height[o]["实测株高 (cm)"],
                height[o]["模拟株高 (cm)"],
            )
        )
        / height[o]["实测株高 (cm)"].mean(),
        np.sqrt(
            mean_squared_error(
                height[~o]["实测株高 (cm)"],
                height[~o]["模拟株高 (cm)"],
            )
        )
        / height[~o]["实测株高 (cm)"].mean(),
    ),
    transform=ax.transAxes,
)
ax.set_xlim(20, 110)
ax.set_ylim(20, 110)
sns.scatterplot(
    x="实测株高 (cm)",
    y="模拟株高 (cm)",
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
        "plant_weight": "模拟地上总干物质 (kg/ha)",
        "Above ground biomass (kg/ha)": "实测地上总干物质 (kg/ha)",
    }
)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                tagb[o]["实测地上总干物质 (kg/ha)"],
                tagb[o]["模拟地上总干物质 (kg/ha)"],
            )
        )
        / tagb[o]["实测地上总干物质 (kg/ha)"].mean(),
        np.sqrt(
            mean_squared_error(
                tagb[~o]["实测地上总干物质 (kg/ha)"],
                tagb[~o]["模拟地上总干物质 (kg/ha)"],
            )
        )
        / tagb[~o]["实测地上总干物质 (kg/ha)"].mean(),
    ),
    transform=ax.transAxes,
)
sns.scatterplot(
    x="实测地上总干物质 (kg/ha)",
    y="模拟地上总干物质 (kg/ha)",
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
            "main_stem_nodes": "模拟主茎节点",
            "Mainstem nodes": "实测主茎节点",
        }
    )
)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls="--", c="k")
ax.text(
    0.55,
    0.05,
    "原版: {:.2%}\n改版: {:.2%}".format(
        np.sqrt(
            mean_squared_error(
                nodes[o]["实测主茎节点"],
                nodes[o]["模拟主茎节点"],
            )
        )
        / nodes[o]["实测主茎节点"].mean(),
        np.sqrt(
            mean_squared_error(
                nodes[~o]["实测主茎节点"],
                nodes[~o]["模拟主茎节点"],
            )
        )
        / nodes[~o]["实测主茎节点"].mean(),
    ),
    transform=ax.transAxes,
)
sns.scatterplot(
    x="实测主茎节点",
    y="模拟主茎节点",
    style="LI method",
    color="k",
    data=nodes,
    ax=ax,
)
plt.legend(bbox_to_anchor=(-0.3, -0.17), loc="upper left", borderaxespad=0)
plt.subplots_adjust(hspace=0.22)
plt.savefig("1v1.png")

#%%
li_truth = (
    li.stack()
    .reset_index()
    .rename(columns={"level_0": "date", "level_1": "treatment", 0: "光能截获率"})
)
li_truth["treatment"] = li_truth["treatment"].str.replace("W", "T")
query_string = "date > '2020-01-01' and date < '2020-10-01' and treatment == 'T5'"
bio["主茎节点"] = bio["Mainstem nodes"].fillna(0)
measured = (
    bio.merge(li_truth, how="outer", on=["date", "treatment"])
    .merge(
        lai.rename(columns={"LAI": "叶面积指数"}), how="outer", on=["date", "treatment"]
    )
    .query(query_string)
    .rename(
        columns={
            "treatment": "Treatment",
            "Above ground biomass (kg/ha)": "地上总干物质的量 (kg/ha)",
            "叶干重(g)": "叶干物质的量 (kg/ha)",
            "主茎干重(g)": "茎干物质的量 (kg/ha)",
            "铃干重(g)": "铃干物质的量 (kg/ha)",
            "叶柄干重(g)": "叶柄干物质的量 (kg/ha)",
            "蕾干重(g)": "蕾干物质的量 (kg/ha)",
            "Plant height (cm)": "株高 (cm)",
        }
    )
)
measured["DOY"] = measured["date"].map(attrgetter("day_of_year"))
data = df.query(query_string)
data["DOY"] = data["date"].map(attrgetter("day_of_year"))
data = data.rename(
    columns={
        "light_interception": "光能截获率",
        "version": "LI method",
        "plant_height": "株高 (cm)",
        "plant_weight": "地上总干物质的量 (kg/ha)",
        "main_stem_nodes": "主茎节点",
        "leaf_weight": "叶干物质的量 (kg/ha)",
        "stem_weight": "茎干物质的量 (kg/ha)",
        "leaf_area_index": "叶面积指数",
        "number_of_squares": "蕾数",
        "boll_weight": "铃干物质的量 (kg/ha)",
        "petiole_weight": "叶柄干物质的量 (kg/ha)",
        "square_weight": "蕾干物质的量 (kg/ha)",
    }
)
data["铃数"] = data["number_of_green_bolls"] + data["number_of_open_bolls"]


def plot_ts(y, ax=None, pred=data, true=measured, bottom=False):
    sns.lineplot(
        x="DOY",
        y=y,
        style="LI method",
        color="k",
        data=pred,
        legend=False,
        ax=ax,
    )
    sns.scatterplot(
        x="DOY",
        y=y,
        color="k",
        data=true,
        ax=ax,
    )
    ax.set_ylabel(y, visible=False)
    ax.text(
        0.05,
        0.75,
        y.replace(" (", "\n("),
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    if not bottom:
        # plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0, labelsize=0)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel("DOY", visible=False)
    else:
        ax.tick_params(direction="in")
        # ax.set_xticks([0, 0.25, 0.5, 0.75, 1])


fig = plt.figure(figsize=(12, 9), dpi=800)
gs = gridspec.GridSpec(4, 3, figure=fig)
ax00 = plt.subplot(gs[0, 0])
plot_ts("光能截获率", ax=ax00)
plot_ts("叶面积指数", ax=plt.subplot(gs[0, 1], sharex=ax00))
plot_ts("主茎节点", ax=plt.subplot(gs[0, 2], sharex=ax00))
plot_ts("株高 (cm)", plt.subplot(gs[1, 0], sharex=ax00))
plot_ts("蕾数", ax=plt.subplot(gs[1, 1], sharex=ax00))
plot_ts("铃数", ax=plt.subplot(gs[1, 2], sharex=ax00))
plot_ts("地上总干物质的量 (kg/ha)", ax=plt.subplot(gs[2, 0], sharex=ax00))
plot_ts("铃干物质的量 (kg/ha)", ax=plt.subplot(gs[2, 1], sharex=ax00))
plot_ts("蕾干物质的量 (kg/ha)", ax=plt.subplot(gs[2, 2], sharex=ax00))
plot_ts("茎干物质的量 (kg/ha)", plt.subplot(gs[3, 0], sharex=ax00), bottom=True)
ax31 = plt.subplot(gs[3, 1], sharex=ax00)
plot_ts("叶干物质的量 (kg/ha)", ax=ax31, bottom=True)
plot_ts("叶柄干物质的量 (kg/ha)", ax=plt.subplot(gs[3, 2], sharex=ax00), bottom=True)
custom_handles = [
    Line2D([0], [0], color="k", label="原版"),
    Line2D([0], [0], linestyle="--", color="k", label="改版"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="实测",
        markerfacecolor="k",
        markersize=8,
    ),
]
ax31.legend(
    handles=custom_handles,
    bbox_to_anchor=(0.5, -0.3),
    ncol=3,
    loc="center",
    borderaxespad=0,
)
fig.subplots_adjust(hspace=0.05)

plt.savefig("time-series.png")

#%%
d = pd.DataFrame(
    data[[f"swc{i}-{j}" for i in range(4) for j in (10, 20, 30)]].stack().reset_index(1)
).rename(columns={0: "Water content (%)"})
d["Water content (%)"] *= 100
d["Position"] = d["level_1"].str.slice(3, 4).astype(int) + 1
d["Depth (cm)"] = d["level_1"].str.slice(5, 7)
d["DOY"] = data["DOY"]
ax = sns.lineplot(
    x="DOY",
    y="Water content (%)",
    style="Depth (cm)",
    err_style=None,
    color="k",
    data=d.reset_index().query("Position == 1"),
)
wc = pd.read_excel("实测.xlsx", sheet_name="土钻含水量").rename(
    columns={
        "日期": "date",
        "处理": "treatment",
    }
)
wc["treatment"] = wc["treatment"].str.replace("W", "T")
wc.index = pd.MultiIndex.from_frame(wc[["date", "treatment"]])
wc = pd.DataFrame(wc.drop(["date", "treatment"], axis=1).stack().reset_index()).rename(
    columns={0: "Water content (%)"}
)
wc["Depth (cm)"] = wc["level_2"].str.slice(2, 4)
wc["Position"] = wc["level_2"].str.slice(0, 1).astype(int)
wc.drop("level_2", axis=1, inplace=True)
wc["DOY"] = wc["date"].map(attrgetter("day_of_year"))
wc["Water content (%)"] *= 1.618
sns.scatterplot(
    x="DOY",
    y="Water content (%)",
    style="Depth (cm)",
    color="k",
    data=wc.query("Position == 1 and treatment == 'T6' and date > '2020-01-01'"),
    ax=ax,
)
