#!/home/tcztzy/.pyenv/versions/thesis/bin/python
import argparse
import datetime
import pathlib
import tarfile
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_pandas
import seaborn as sns
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from datav import DataVGeoAtlasFeature
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy

if TYPE_CHECKING:
    from typing import Generator

SimSun = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")


def research_site(fontproperties: FontProperties, path, dpi=800):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 7))
    main_box = (80.5, 82, 40.25, 41)
    ax_main = plt.subplot(1, 1, 1, projection=proj)
    ax_main.set_extent(main_box, crs=proj)

    ax_main.add_feature(DataVGeoAtlasFeature(659002, facecolor="darkgreen", alpha=0.2))
    ax_main.add_feature(cfeature.LAKES)
    ax_main.add_feature(cfeature.RIVERS)
    site = (81.196, 40.624)
    ax_main.plot(
        *site,
        marker="o",
        color="green",
        markersize=10,
        transform=ccrs.Geodetic(),
    )
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax_main)
    text_transform = offset_copy(geodetic_transform, units="dots", x=-25)
    ax_main.text(
        site[0],
        site[1] + 0.03,
        "试验站",
        transform=text_transform,
        fontproperties=fontproperties,
    )
    xmain = np.linspace(*main_box[:2], 4)
    ymain = np.linspace(*main_box[2:], 4)
    ax_main.gridlines(xlocs=xmain, ylocs=ymain, linestyle=":")
    # set custom formatting for the tick labels
    ax_main.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax_main.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax_main.yaxis.tick_right()
    ax_main.set_xticks(xmain, crs=ccrs.PlateCarree())
    ax_main.set_yticks(ymain, crs=ccrs.PlateCarree())

    ax_inset = fig.add_axes([0, 0.6, 0.3, 0.35], projection=proj)
    ax_inset.set_extent((73, 136, 2, 51))
    ax_inset.add_feature(
        DataVGeoAtlasFeature(
            100000, full=True, facecolor="lightgreen", edgecolor="gray"
        )
    )
    ax_inset.add_feature(DataVGeoAtlasFeature(650000, facecolor="limegreen"))
    # ax_inset.add_feature(DataVGeoAtlasFeature(652900, facecolor="green", edgecolor="none"))
    ax_inset.add_feature(
        DataVGeoAtlasFeature(659002, facecolor="darkgreen", edgecolor="black")
    )
    xinset = np.linspace(73, 136, 5)
    yinset = np.linspace(2, 51, 5)
    ax_inset.gridlines(xlocs=xinset, ylocs=yinset, linestyle=":")
    ax_inset.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax_inset.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax_inset.xaxis.tick_top()
    # set inset tick labels
    ax_inset.set_xticks(xinset, crs=ccrs.PlateCarree())
    ax_inset.set_yticks(yinset, crs=ccrs.PlateCarree())

    plt.savefig(path, dpi=dpi)


def alar_gsod() -> pd.DataFrame:
    def g() -> "Generator[pd.DataFrame]":
        for year in (2019, 2020, 2021):
            with tarfile.open(f"{year}.tar.gz", "r:gz") as tar:
                yield pd.read_csv(
                    tar.extractfile("51730099999.csv"),
                    na_values=["999.9", "99.99", "9999.9"],
                    parse_dates=[1],
                    index_col=1,
                )

    return pd.concat(g())


def f2c(d):
    return (d - 32) / 1.8


def plot_climate_one_year(data, ax, fontproperties=SimSun):
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines["right"].set_position(("axes", 1.05))
    ax.set_ylim(0, 45)
    twin1.set_ylim(-5, 40)
    twin2.set_ylim(0, 20)
    (p1,) = ax.plot(
        data["DOY"],
        data["ALLSKY_SFC_SW_DWN"],
        color="gray",
        linestyle="--",
        label="SRAD",
    )
    p2_color = cm.hot(0.3)

    sns.regplot(
        x="DOY",
        y="MAX",
        line_kws={"color": p2_color},
        scatter_kws={"color": p2_color},
        data=data,
        ax=twin1,
        lowess=True,
        label="TMAX",
    )
    p2 = twin1.get_children()[0]
    p3_color = cm.hot(0.0)
    sns.regplot(
        x="DOY",
        y="MIN",
        line_kws={"color": p3_color},
        scatter_kws={"color": p3_color},
        data=data,
        ax=twin1,
        lowess=True,
        label="TMIN",
    )
    p3 = twin1.get_children()[1]
    p4 = twin2.bar(
        data["DOY"],
        data["PRCP"],
        color="deepskyblue",
        label="RAIN",
    )

    ax.set_ylabel("太阳辐射 ($\mathrm{MJ\ m^{-3} d^{-1}}$)", fontproperties=fontproperties)
    twin1.set_ylabel("气温 ($\mathrm{C^{\circ}}$)", fontproperties=fontproperties)
    twin2.set_ylabel("降水 (mm)", fontproperties=fontproperties)
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2_color)
    twin2.yaxis.label.set_color("deepskyblue")

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
    # twin1.tick_params(axis="y", colors=p2_color, **tkw)

    yticklabels = twin1.get_yticklabels()
    for ticklabel, tickcolor in zip(
        yticklabels,
        cm.hot(np.linspace(0.0, 0.3, len(yticklabels))),
    ):
        ticklabel.set_color(tickcolor)
    twin2.tick_params(axis="y", colors="deepskyblue", **tkw)
    ax.tick_params(axis="x", **tkw)
    ax.legend(handles=[p1, p2, p3, p4])


def plot_climate(data, path="climate.png"):
    _, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=800)

    for year, ax in zip((2019, 2020, 2021), axes):
        plot_climate_one_year(data.loc[f"{year}-04-01":f"{year}-11-01"], ax)
        if year < 2021:
            ax.tick_params(
                axis="x",
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )
    axes[2].set_xlabel("DOY")
    plt.subplots_adjust(hspace=0.1)

    plt.savefig(path)


if __name__ == "__main__":
    pint_pandas.PintType.ureg.setup_matplotlib()
    ureg = pint.UnitRegistry()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", default=False, action="store_true")
    args = parser.parse_args()
    force: bool = args.force
    if not (research_site_path := pathlib.Path("research_site.png")).exists() or force:
        research_site(fontproperties=SimSun, path=research_site_path)
    if not (alar_climate_path := pathlib.Path("alar.climate.pkl")).exists() or force:
        power = pd.read_csv(
            "POWER_Point_Daily_20190101_20220319_040d6240N_081d1960E_LST.csv",
            skiprows=19,
        )
        power.index = power["YEAR"].apply(lambda y: datetime.date(y, 1, 1)) + power[
            "DOY"
        ].apply(lambda doy: datetime.timedelta(days=doy - 1))
        alar_climate = alar_gsod().merge(power, left_index=True, right_index=True)
        alar_climate.to_pickle(alar_climate_path)
    else:
        alar_climate = pd.read_pickle(alar_climate_path)
    alar_climate = alar_climate.astype(
        {
            "MAX": "pint[degF]",
            "MIN": "pint[degF]",
            "WDSP": "pint[knot]",
            "PRCP": "pint[inch]",
        }
    )
    for col, to in {
        "MAX": "degC",
        "MIN": "degC",
        "WDSP": "km/d",
        "PRCP": "mm",
    }.items():
        alar_climate[col] = alar_climate[col].pint.to(to)
    alar_climate.loc[alar_climate["PRCP"] == 9.906 * ureg.mm, "PRCP"] = np.nan
    if not (climate_path := pathlib.Path("climate.png")).exists() or force:
        plot_climate(alar_climate, climate_path)
