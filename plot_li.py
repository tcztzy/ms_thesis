import collections
import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from scipy.ndimage.filters import gaussian_filter

light_interception_distribution = collections.defaultdict(dict)
light_interception = collections.defaultdict(dict)
SimSun = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")


def calc_light_interception(result, date, treatment):
    max_value = result.iloc[-1].mean()
    min_value = result.iloc[0].mean()
    light_interception_distribution[date][treatment] = result.apply(
        lambda x: (max_value - x) / max_value
    )
    light_interception[date][treatment] = (max_value - min_value) / max_value


for xlsx in pathlib.Path("光分布数据/处理后数据").glob("**/*.xlsx"):
    y = int(xlsx.parent.name[:-1])
    m, d = map(int, xlsx.stem.split("."))
    date = datetime.date(y, m, d)
    if date.year == 2020:
        df = pd.read_excel(xlsx, index_col=(0, 1))
        data = (
            df.rename(columns={"Unnamed: 2": "类型"})
            .rename(
                index={
                    "第一层": 0,
                    "第二层": 1,
                    "第三层": 2,
                    "第四层": 3,
                    "第五层": 4,
                    "第六层": 5,
                },
                columns={f"点位{i + 1}": i for i in range(5)},
            )[["类型"] + list(range(5))]
            .sort_index()
        )
        data.loc[("W2", 1), 2] = (2200, 328.43)  # trick

        data = data[data["类型"] == "入射光"][range(5)] - data[data["类型"] == "反射光"][range(5)]
        for i in range(6):
            treatment = f"W{i + 1}"
            calc_light_interception(data.loc[treatment], date, treatment)
            result = light_interception_distribution[date][treatment]
            if result.shape[0] < 6:
                for j in range(result.shape[0], 6):
                    result.loc[j] = 0
    elif date == datetime.date(2019, 7, 11):
        df = (
            pd.read_excel(
                xlsx,
                skiprows=(1, 2),
                index_col=(0, 1),
            )
            .rename(
                index={
                    "第一层": 0,
                    "第二层": 1,
                    "第三层": 2,
                    "第四层": 3,
                    "第五层": 4,
                    "第六层": 5,
                },
                columns={f"点位{i + 1}": i for i in range(5)},
            )
            .sort_index()
        )
        for i in range(6):
            treatment = f"W{i + 1}"
            result = (
                df[[treatment, f"Unnamed: {3 * i + 3}", f"Unnamed: {3 * i + 4}"]]
                .T.mean()
                .unstack()
            )
            calc_light_interception(result, date, treatment)
    else:
        dfs = pd.read_excel(
            xlsx,
            index_col=(0, 1),
            skiprows=(0,),
            sheet_name=[f"W{i + 1}" for i in range(6)],
        )
        for treatment, df in dfs.items():
            data = df.rename(
                index={
                    "第一层": 0,
                    "第二层": 1,
                    "第三层": 2,
                    "第四层": 3,
                    "第五层": 4,
                    "第六层": 5,
                    **{f"点位{i + 1}": i for i in range(5)},
                },
            ).sort_index()
            data[data == 1] = None
            time = datetime.datetime.combine(
                date,
                datetime.time(16, 0, 0),
            )
            t = min(
                data.columns,
                key=lambda t: abs(datetime.datetime.combine(date, t) - time),
            )
            calc_light_interception(data[t].unstack(), date, treatment)


def plot_light_interception(
    data, title=None, xy=(80, 100), samples=(5, 6), scale=100, sigma=0.5, ax=None
):
    if ax is None:
        ax = plt.gca()
    x_samples, y_samples = samples
    x, y = xy
    x_step = x / (x_samples * scale - 1)
    y_step = y / (y_samples * scale - 1)
    x = np.arange(0, x + 0.1 * x_step, x_step)
    y = np.arange(0, y + 0.1 * y_step, y_step)
    X, Y = np.meshgrid(x, y)
    Z = scipy.ndimage.zoom(
        gaussian_filter(data, sigma) if sigma is not None else data,
        scale,
    )
    CS = ax.contourf(
        X,
        Y,
        Z,
        levels=np.arange(0, 1.05, 0.05),
        cmap=cm.YlOrRd,
    )
    ax.clabel(CS, levels=CS.levels[4::5], colors="black", inline=False, fontsize=10)
    plt.colorbar(CS)
    ax.set_title(title)


def plot_li_dist(data, fontproperties, ax, t):
    data = data[["lai{:02}".format(i) for i in range(20)] + ["date"]]
    data.index = data["date"].apply(
        lambda x: (
            datetime.datetime.strptime(x, "%Y-%m-%d").date() - datetime.date(2019, 5, 7)
        ).days
        + 1
    )
    data.drop("date", axis=1, inplace=True)
    data.columns = range(0, 100, 5)
    sns.heatmap(
        data.T.drop([-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0], axis=1),
        xticklabels=28,
        yticklabels=4,
        cbar_kws={"label": "LAI"},
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_ylabel("株高 (cm)", fontproperties=fontproperties)
    if t < 4:
        ax.set_xlabel("")
        ax.tick_params(
            axis="x",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
    else:
        ax.set_xlabel("DAE", fontproperties=fontproperties)
    ax.title.set_text(f"W{t+1}")


if __name__ == "__main__":
    df = pd.read_csv("result_demo.csv", index_col=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for t in range(6):
        plot_li_dist(
            df.query(
                f"version == 'Modified' and treatment == 'T{t + 1}' and date < '2020-01-01'"
            ),
            SimSun,
            axes[t // 2, t % 2],
            t
        )
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("lai_dist_2019.png", dpi=800)
