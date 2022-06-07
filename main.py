# %%
import multiprocessing as mp
import pathlib
import shutil
import subprocess
import tempfile

import pandas as pd
import toml
from SALib.analyze import sobol
from SALib.sample import saltelli

C2K_EXE = r"C:\Users\Tang\Documents\cotton2k\target\release\cotton2k.exe"


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


def generate_profile(sample, year, i, v=0):
    profile = toml.load(f"ALAR{year}.{i}.toml")
    profile["cultivar_parameters"] = [float(p) for p in sample[:50]]
    if v == 2:
        profile["light_intercept_method"] = v
        profile["light_intercept_parameters"] = [float(p) for p in sample[50:]]
    return profile


def run(sample, year, treatment, v=0):
    with tempfile.TemporaryDirectory() as workspace:
        workspace = pathlib.Path(workspace)
        with (workspace / "ALAR.TOML").open("w") as f:
            toml.dump(generate_profile(sample, year, treatment, v), f)
        shutil.copyfile("soil_imp.csv", workspace / "soil_imp.csv")
        c = climate.loc[114:297] if year == 2019 else climate.loc[480:663]
        c.to_csv(workspace / "ALAR.CSV", index=False)
        subprocess.run([C2K_EXE, str(workspace / "ALAR.TOML"), "ALAR"], cwd=workspace, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return pd.read_csv(workspace / "output.csv", parse_dates=[0])


def safe_run(args):
    """Call run(), catch exceptions."""
    sample_id, sample, year, treatment, v = args
    filepath = (
        pathlib.Path("sa_eval_result") / f"{year}.{treatment}.{v}.{sample_id}.csv"
    )
    if not filepath.exists():
        try:
            run(sample, year, treatment, v).to_csv(filepath)
        except Exception as e:
            print("error: %s run(*%r)" % (e, (sample_id, sample, year, treatment, v)))


def sa_eval(o_samples, m_samples):
    # start processes
    pool = mp.Pool()  # use all available CPUs
    pool.map(
        safe_run,
        (
            (i, s, y, t, 2)
            for i, s in enumerate(m_samples)
            for y in (2019, 2020)
            for t in range(6)
        ),
    )
    pool.map(
        safe_run,
        (
            (i, s, y, t)
            for i, s in enumerate(o_samples)
            for y in (2019, 2020)
            for t in range(6)
        ),
    )


def sensitivity_analysis(number_of_samples=16):
    if not pathlib.Path("sa_eval_result.csv").exists():
        Y = sa_eval(
            saltelli.sample(o_problem, number_of_samples),
            saltelli.sample(m_problem, number_of_samples),
        )
    else:
        Y = pd.read_csv("sa_eval_result.csv", index_col=0)

    def g():
        for year in (2019, 2020):
            for t in range(6):
                for method in (0, 2):
                    yield sobol.analyze(
                        o_problem,
                        Y.query(
                            f"date >= '{year}-01-01' and "
                            f"date < '{year + 1}-01-01' and "
                            f"treatment == 't{t + 1}' and "
                            f"method == {method}"
                        ),
                    )["S1"]

    return pd.DataFrame(g())
