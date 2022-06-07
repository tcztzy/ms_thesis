#%%
import pathlib
import pickle

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.decomposition.asf import ASF
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_reference_directions,
    get_sampling,
    get_termination,
)
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error

from main import run

#%%
li = pd.read_excel("LI_MEAN.xlsx", index_col=0)
土钻含水量 = pd.read_excel("实测.xlsx", sheet_name="土钻含水量")
bio = pd.read_excel("BIO_FINAL.xlsx", index_col=0)
lai = pd.read_excel("LAI_FINAL.xlsx", index_col=0)
seedcotton_yield = pd.read_excel("seedcotton_yield.xlsx", index_col=[0, 1])[0]
sa_problem = pd.read_csv("sa.csv")


def all_error(df):
    lai_pred = df[df["date"].isin(lai["date"])].sort_values(["date", "treatment"])
    li_pred = df[df["date"].isin(li.index)].sort_values(["date", "treatment"])
    bio_pred = df[df["date"].isin(bio["date"])].sort_values(["date", "treatment"])
    scy_pred = df.query("date == '2019-10-25' or date == '2020-10-25'")[
        "seed_cotton_yield"
    ]
    scy_error = (
        (
            np.sqrt(
                mean_squared_error(
                    seedcotton_yield,
                    scy_pred,
                )
            )
            / seedcotton_yield.mean()
        )
        if len(scy_pred) == len(seedcotton_yield)
        else 100
    )
    lai_err = (
        (
            np.sqrt(mean_squared_error(lai["LAI"], lai_pred["leaf_area_index"]))
            / lai["LAI"].mean()
        )
        if len(lai) == len(lai_pred)
        else 100
    )
    li_err = (
        (
            np.sqrt(mean_squared_error(li.stack(), li_pred["light_interception"]))
            / li.stack().mean()
        )
        if len(li.stack()) == len(li_pred)
        else 100
    )
    z_err = (
        (
            np.sqrt(
                mean_squared_error(bio["Plant height (cm)"], bio_pred["plant_height"])
            )
            / bio["Plant height (cm)"].mean()
        )
        if len(bio) == len(bio_pred)
        else 100
    )
    agb_err = (
        (
            np.sqrt(
                mean_squared_error(
                    bio["Above ground biomass (kg/ha)"], bio_pred["plant_weight"]
                )
            )
            / bio["Above ground biomass (kg/ha)"].mean()
        )
        if len(bio) == len(bio_pred)
        else 100
    )
    node_err = (
        (
            np.sqrt(
                mean_squared_error(
                    bio["Mainstem nodes"].fillna(0), bio_pred["main_stem_nodes"]
                )
            )
            / bio["Mainstem nodes"].fillna(0).mean()
        )
        if len(bio) == len(bio_pred)
        else 100
    )
    return (
        scy_error,
        lai_err,
        li_err,
        z_err,
        agb_err,
        node_err,
    )


sv = np.append(
    np.array(
        [1, 4, 12, 15, 21, 22, 26, 27, 30, 31, 32, 34, 35, 41, 42, 43, 47, 48, 49, 50]
    )
    - 1,
    np.array(range(50, 70)),
)
iv = np.array(
    [
        np.nan,
        0.3,
        0.014,
        np.nan,
        1.6,
        0.010,
        24.0,
        0.10,
        28,
        0.3293,
        8.8,
        np.nan,
        0.040,
        0.014,
        np.nan,
        1.5,
        0.40,
        0.140,
        0.20,
        0.02,
        np.nan,
        np.nan,
        0.10,
        0.175,
        2.20,
        np.nan,
        np.nan,
        2.15,
        1.36,
        np.nan,
        np.nan,
        np.nan,
        0.80,
        np.nan,
        np.nan,
        -54.00,
        0.80,
        3.20,
        -292.0,
        1.08,
        np.nan,
        np.nan,
        np.nan,
        0.48,
        0.24,
        0.08,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=40,
            n_obj=6,
            xl=sa_problem["LB"][sv].values,
            xu=sa_problem["UB"][sv].values,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        all_result = pd.DataFrame(
            columns="date, light_interception, lint_yield, leaf_area_index, seed_cotton_yield, plant_height, main_stem_nodes, leaf_weight, stem_weight, number_of_green_bolls, number_of_open_bolls, boll_weight, plant_weight, swc0-10, swc0-20, swc0-30, swc1-10, swc1-20, swc1-30, swc2-10, swc2-20, swc2-30, swc3-10, swc3-20, swc3-30, treatment".split(
                ", "
            )
        )
        X = np.append(iv.copy(), np.ones(20) * -3)
        for i, v in zip(sv, x):
            X[i] = v
        for year in (2019, 2020):
            for i in range(6):
                result = run(X, year, i, 2)
                result["treatment"] = f"W{i + 1}"
                all_result = all_result.append(result, ignore_index=True)
        out["F"] = all_error(all_result)


class OriginalProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=20,
            n_obj=6,
            xl=sa_problem["LB"][sv[:20]].values,
            xu=sa_problem["UB"][sv[:20]].values,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        all_result = pd.DataFrame(
            columns="date, light_interception, lint_yield, leaf_area_index, seed_cotton_yield, plant_height, main_stem_nodes, leaf_weight, stem_weight, number_of_green_bolls, number_of_open_bolls, boll_weight, plant_weight, swc0-10, swc0-20, swc0-30, swc1-10, swc1-20, swc1-30, swc2-10, swc2-20, swc2-30, swc3-10, swc3-20, swc3-30, treatment".split(
                ", "
            )
        )
        X = iv.copy()
        for i, v in zip(sv[:20], x):
            X[i] = v
        for year in (2019, 2020):
            for i in range(6):
                result = run(X, year, i)
                result["treatment"] = f"W{i + 1}"
                all_result = all_result.append(result, ignore_index=True)
        out["F"] = all_error(all_result)


def moo(method=0, pop=None):
    npz = pathlib.Path(f"moo_result_{method}.npz")
    if npz.exists() and pop is None:
        with np.load(str(npz)) as data:
            pop = data["X"]

    algorithm = NSGA3(
        pop_size=128,
        ref_dirs=get_reference_directions("das-dennis", 6, n_partitions=4),
        n_offsprings=10,
        sampling=get_sampling("real_random") if pop is None else pop,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

    problem = MyProblem() if method == 2 else OriginalProblem()

    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 20),
        save_history=True,
        verbose=True,
    )

    with open(f"moo_result_{method}.pkl", "wb") as f:
        pickle.dump(res, f)

    with open(f"moo_result_{method}.npz", "wb") as f:
        np.savez(f, X=res.X, F=res.F)


def pick(F, choice="ASF", weights=np.array([1, 1, 1, 1, 1, 1])):
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    if choice == "ASF":
        decomp = ASF()
        i = decomp.do(nF, 1 / weights).argmin()
    elif choice == "PseudoWeight":
        i = PseudoWeights(weights).do(nF)
    return i


def main(method=0, weights=np.array([1, 0.00001, 0.00001, 0.00001, 1, 1])):
    with open(f"moo_result_{method}.pkl", "rb") as f:
        res = pickle.load(f)
    X = res.X[pick(res.F, weights=weights)]
    print(res.F[pick(res.F, weights=weights)])
    return X


if __name__ == "__main__":
    #moo(2)
    main()
    main(
        2,
        # weights=np.array([6, 1, 1, 1, 6, 3]),
    )
    with open("moo_result_0.pkl", "rb") as f:
        res = pickle.load(f)
    X0 = res.X[9]
    with open("moo_result_2.pkl", "rb") as f:
        res = pickle.load(f)
    X2 = res.X[25]
    print(X0)
    print(X2)
