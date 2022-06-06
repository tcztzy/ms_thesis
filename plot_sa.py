#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from SALib.analyze import sobol

mpl.rc("font", family="Microsoft YaHei")
climate = pd.read_csv("climate.csv")

sa_problem = pd.read_csv("sa.csv")
m_problem = {
    "num_vars": len(sa_problem),
    "names": sa_problem["Parameter"].values,
    "bounds": sa_problem[["LB", "UB"]].values,
}
o_problem = {
    "num_vars": 50,
    "names": sa_problem["Parameter"].values[:50],
    "bounds": sa_problem[["LB", "UB"]].values[:50],
}


def sensitivity_analysis():
    Y = pd.read_csv("sa_eval_result.csv", index_col=0)

    def S():
        for method in (0, 2):
            for metric, y in dict(
                株高=Y.query(
                    f"date == '2020-07-01' and "
                    f"treatment == 4 and "
                    f"method == {method}"
                )["plant_height"].values,
                叶干物质的量=Y.query(
                    f"date == '2020-05-25' and "
                    f"treatment == 4 and "
                    f"method == {method}"
                )["leaf_weight"].values,
                皮棉产量=Y.query(
                    f"date == '2020-09-25' and "
                    f"treatment == 4 and "
                    f"method == {method}"
                )["lint_yield"].values,
                光能截获率=Y.query(
                    f"date == '2020-06-25' and "
                    f"treatment == 4 and "
                    f"method == {method}"
                )["light_interception"].values,
            ).items():
                sa_result = sobol.analyze(
                    o_problem if method == 0 else m_problem,
                    y,
                )
                for v in range(50):
                    yield (
                        f"VARPAR{v+1:02}",
                        np.abs(sa_result["S1"][v]),
                        "一阶",
                        "原版" if method == 0 else "改版",
                        metric,
                    )
                    yield (
                        f"VARPAR{v+1:02}",
                        np.abs(sa_result["ST"][v]),
                        "全局",
                        "原版" if method == 0 else "改版",
                        metric,
                    )
                if method == 2:
                    for v in range(20):
                        yield (
                            f"LIPAR{v+1:02}",
                            np.abs(sa_result["S1"][v + 50]),
                            "一阶",
                            "原版" if method == 0 else "改版",
                            metric,
                        )
                        yield (
                            f"LIPAR{v+1:02}",
                            np.abs(sa_result["ST"][v + 50]),
                            "全局",
                            "原版" if method == 0 else "改版",
                            metric,
                        )

    df = pd.DataFrame(S())
    df.columns = ["name", "value", "敏感性", "version", "metric"]
    return df


if __name__ == "__main__":
    sa = sensitivity_analysis()
    for metric in ("株高", "叶干物质的量", "皮棉产量", "光能截获率"):
        for version in ("原版", "改版"):
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax = sns.barplot(
                data=sa.query(f"metric == '{metric}' and version == '{version}'"),
                x="name",
                y="value",
                hue="敏感性",
            )
            length = 50 if version == "原版" else 70
            ax.set_ylim((1e-3, 1))
            ax.plot((-1, length), (0.05, 0.05), ls="--", c="k")
            ax.annotate(
                "0.05",
                (length, 0.05),
                xytext=(length + 2, 0.06),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle3"),
            )
            ax.plot((-1, length), (0.1, 0.1), ls="--", c="k")
            ax.annotate(
                "0.10",
                (length, 0.1),
                xytext=(length + 2, 0.12),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle3"),
            )
            ax.set_xlabel("")
            ax.set_ylabel("一阶和全局敏感性")
            ax.set_title(metric + " " + version)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            hatches = ("x", "///")
            for i, the_bar in enumerate(ax.patches):
                the_bar.set_hatch(hatches[i // length])
            ax.legend()
            plt.xticks(rotation=-60)
            plt.yscale("log")
            plt.savefig(metric + " " + version + ".png", dpi=800)
