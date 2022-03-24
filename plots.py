import pathlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tarfile
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from datav import DataVGeoAtlasFeature
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy

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


def alar_gsod():
    def g():
        for year in (2019, 2020, 2021):
            with tarfile.open(f"{year}.tar.gz", "r:gz") as tar:
                yield pd.read_csv(
                    tar.extractfile("51730099999.csv"),
                    na_values=["999.9", "99.99", "9999.9"],
                    parse_dates=[1],
                    index_col=1,
                )

    return pd.concat(g())


if __name__ == "__main__":
    if not (research_site_path := pathlib.Path("research_site.png")).exists():
        research_site(fontproperties=SimSun, path=research_site_path)
    if not (gsod_path := pathlib.Path("alar.gsod.csv")).exists():
        alar_climate = alar_gsod()
        alar_climate.to_csv(gsod_path)
    else:
        alar_climate = pd.read_csv(gsod_path, index_col=0, parse_dates=[0])
    print(alar_climate)
