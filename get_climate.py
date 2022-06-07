import pandas as pd

power = pd.read_csv(
    r"C:\Users\tcztz\Downloads\POWER_Point_Daily_20190101_20220319_040d6240N_081d1960E_LST.csv",
    skiprows=22,
).query("YEAR == 2020 or YEAR == 2019")
gsod = pd.read_csv(
    r"C:\Users\Tang\Downloads\2019-01-01 2019-12-31 ALAR CN.csv",
    na_values=["999.9", "99.99", "9999.9"],
    parse_dates=[1],
)
gsod = gsod.append(
    pd.read_csv(
        r"C:\Users\Tang\Downloads\2020-01-01 2020-12-31 ALAR CN.csv",
        na_values=["999.9", "99.99", "9999.9"],
        parse_dates=[1],
    ),
    ignore_index=True,
)
gsod["irradiation"] = power["ALLSKY_SFC_SW_DWN"]
gsod["TMIN"] = gsod["MIN"].map(lambda x: (x - 32) / 1.8)  # deg F to deg C
gsod["TMAX"] = gsod["MAX"].map(lambda x: (x - 32) / 1.8)
gsod["TDEW"] = power["T2MDEW"]
gsod["WIND"] = gsod["WDSP"] * 0.514444 * 86400 / 1000  # knot to km/d
gsod["RAIN"] = (gsod["PRCP"] * 25.4).fillna(0)  # inch to mm
gsod.rename(columns=str.lower)[
    ["date", "irradiation", "tmax", "tmin", "rain", "wind", "tdew"]
].to_csv("climate.csv", index=False)
