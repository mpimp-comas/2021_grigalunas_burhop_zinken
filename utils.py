#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
from typing import List, Dict, Set, Callable, Union

# import sys

import pandas as pd
from pandas import DataFrame
import numpy as np
import itertools
from scipy import stats

try:
    from pandarallel import pandarallel

    PARALLEL = True
except ImportError:
    PARALLEL = False


try:
    from tqdm.notebook import tqdm

    tqdm.pandas()
    TQDM = True
except ImportError:
    TQDM = False

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Desc
import rdkit.Chem.rdMolDescriptors as rdMolDesc
from rdkit.Chem import Fragments
from rdkit.Chem import Crippen

# from rdkit.Chem.MolStandardize import rdMolStandardize
# from rdkit.Chem.MolStandardize.validate import Validator
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
from rdkit.Chem.MolStandardize.standardize import Standardizer
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from rdkit import rdBase

rdBase.DisableLog("rdApp.info")
# rdBase.DisableLog("rdApp.warn")

molvs_s = Standardizer()
molvs_l = LargestFragmentChooser()
molvs_u = Uncharger()
molvs_t = TautomerCanonicalizer(max_tautomers=100)


PALETTE = [
    "#EB5763",  # red
    "#47ED47",  # green
    "#81C0EB",  # blue
    "#FDD247",  # orange
]

# from sklearn import decomposition
# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns


def get_value(str_val: str):
    """convert a string into float or int, if possible."""
    if not str_val:
        return np.nan
    try:
        val = float(str_val)
        if "." not in str_val:
            val = int(val)
    except ValueError:
        val = str_val
    return val


def read_sdf(fn, merge_prop: str = None, merge_list: Union[List, Set] = None):
    """Create a DataFrame instance from an SD file.
    The input can be a single SD file or a list of files and they can be gzipped (fn ends with `.gz`).
    If a list of files is used, all files need to have the same fields.
    The molecules will be converted to Smiles.

    Parameters:
    ===========
    merge_prop: A property in the SD file on which the file should be merge
        during reading.
    merge_list: A list or set of values on which to merge.
        Only the values of the list are kept.
    """

    d = {"Smiles": []}
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol"]}
    if merge_prop is not None:
        ctr["NotMerged"] = 0
    first_mol = True
    sd_props = set()
    if not isinstance(fn, list):
        fn = [fn]
    for f in fn:
        do_close = True
        if isinstance(f, str):
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
        else:
            file_obj = f
            do_close = False
        reader = Chem.ForwardSDMolSupplier(file_obj)
        for mol in reader:
            ctr["In"] += 1
            if not mol:
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                # Is the SD file name property used?
                name = mol.GetProp("_Name")
                if len(name) > 0:
                    has_name = True
                    d["Name"] = []
                else:
                    has_name = False
                for prop in mol.GetPropNames():
                    sd_props.add(prop)
                    d[prop] = []
            if merge_prop is not None:
                # Only keep the record when the `merge_prop` value is in `merge_list`:
                if get_value(mol.GetProp(merge_prop)) not in merge_list:
                    ctr["NotMerged"] += 1
                    continue
            mol_props = set()
            ctr["Out"] += 1
            for prop in mol.GetPropNames():
                if prop in sd_props:
                    mol_props.add(prop)
                    d[prop].append(get_value(mol.GetProp(prop)))
                mol.ClearProp(prop)
            if has_name:
                d["Name"].append(get_value(mol.GetProp("_Name")))
                mol.ClearProp("_Name")

            # append NAN to the missing props that were not in the mol:
            missing_props = sd_props - mol_props
            for prop in missing_props:
                d[prop].append(np.nan)
            d["Smiles"].append(mol_to_smiles(mol))
        if do_close:
            file_obj.close()
    # Make sure, that all columns have the same length.
    # Although, Pandas would also complain, if this was not the case.
    d_keys = list(d.keys())
    if len(d_keys) > 1:
        k_len = len(d[d_keys[0]])
        for k in d_keys[1:]:
            assert k_len == len(d[k]), f"{k_len=} != {len(d[k])}"
    result = pd.DataFrame(d)
    print(ctr)
    return result


def mol_to_smiles(mol: Mol, canonical: bool = True) -> str:
    """Generate Smiles from mol.

    Parameters:
    ===========
    mol: the input molecule
    canonical: whether to return the canonical Smiles or not

    Returns:
    ========
    The Smiles of the molecule (canonical by default). NAN for failed molecules."""

    if mol is None:
        return np.nan
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return np.nan


def smiles_to_mol(smiles: str) -> Mol:
    """Generate a RDKit Molecule from a Smiles.

    Parameters:
    ===========
    smiles: the input string

    Returns:
    ========
    The RDKit Molecule. If the Smiles parsing failed, NAN instead.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        return np.nan
    except ValueError:
        return np.nan


def drop_cols(df: DataFrame, cols: List[str]) -> DataFrame:
    """Remove the list of columns from the dataframe.
    Listed columns that are not available in the dataframe are simply ignored."""
    df = df.copy()
    cols_to_remove = set(cols).intersection(set(df.keys()))
    df = df.drop(cols_to_remove, axis=1)
    return df


def standardize_mol(
    mol: Mol, remove_stereo: bool = False, canonicalize_tautomer: bool = False
):
    """Standardize the molecule structures.
    Returns:
    ========
    Smiles of the standardized molecule. NAN for failed molecules."""

    if mol is None:
        return np.nan
    mol = molvs_l.choose(mol)
    mol = molvs_u.uncharge(mol)
    mol = molvs_s.standardize(mol)
    if remove_stereo:
        mol = molvs_s.stereo_parent(mol)
    if canonicalize_tautomer:
        mol = molvs_t.canonicalize(mol)
    # mol = largest.choose(mol)
    # mol = uncharger.uncharge(mol)
    # mol = normal.normalize(mol)
    # mol = enumerator.Canonicalize(mol)
    return mol_to_smiles(mol)


def apply_to_smiles(
    df: DataFrame,
    smiles_col: str,
    funcs: Dict[str, Callable],
    parallel: bool = False,
    workers: int = 6,
) -> DataFrame:
    """Calculation of chemical properties,
    directly on the Smiles.
    Parameters:
    ===========
    df: Pandas DataFrame
    smiles_col: Name of the Smiles column
    funcs: A dict of names and functions to apply to the mol object.
        The keys are the names of the generated columns,
        the values are the functions.
        If the generation of the intermediary mol object fails, NAN is returned.
    parallel: Set to True when the function should be run in parallel (default: False).
        pandarallel has to be installed for this.
    workers: Number of workers to be used when running in parallel.
    Returns:
    ========
    New DataFrame with the calculated properties.

    Example:
    ========
    `df` is a DataFrame that contains a "Smiles" column.
    >>> from rdkit.Chem import Descriptors as Desc
    >>> df2 = apply_to_smiles(df, "Smiles", {"MW": Desc.MolWt, "LogP": Desc.MolLogP})
    """

    func_items = funcs.items()
    func_keys = {i: x[0] for i, x in enumerate(func_items)}
    func_vals = [x[1] for x in func_items]

    def _apply(smi):
        if not isinstance(smi, str):
            res = [np.nan] * len(func_vals)
            return pd.Series(res)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            res = [np.nan] * len(func_vals)
            return pd.Series(res)
        res = []
        for f in func_vals:
            try:
                r = f(mol)
                res.append(r)
            except:
                res.append(np.nan)
        return pd.Series(res)

    df = df.copy()
    fallback = True
    if parallel:
        if not PARALLEL:
            print("Parallel option not available. Please install pandarallel.")
            print("Using single core calculation.")
        else:
            pandarallel.initialize(nb_workers=workers, progress_bar=TQDM)
            result = df[smiles_col].parallel_apply(_apply)
            fallback = False

    if fallback:
        if TQDM:
            result = df[smiles_col].progress_apply(_apply)
        else:
            result = df[smiles_col].apply(_apply)
    result = result.rename(columns=func_keys)
    df = pd.concat([df, result], axis=1)
    return df


def get_atom_set(mol: Mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def filter_mols(
    df: DataFrame, smiles_col: str, filter: Union[str, List[str]]
) -> DataFrame:
    """Apply different filters to the molecules.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heacy atoms
            - MaxHeavyAtoms: Keep only molecules with 75 or less heacy atoms
            - Duplicates: Remove duplicates by InChiKey
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}

    def has_non_medchem_atoms(mol: Mol):
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(mol: Mol) -> bool:
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    print(f"Applying filters ({len(filter)})...")
    for filt in filter:
        if filt == "Isotopes":
            df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            df = apply_to_smiles(
                df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            )
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms >= 3")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms <= 75")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "Duplicates":
            df = apply_to_smiles(
                df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            )
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    return df


def read_tsv(
    input_tsv: str, smiles_col: str = "Smiles", parse_smiles: bool = False
) -> pd.DataFrame:
    """Read a tsv file, optionnally converting smiles into RDKit molecules.

    Parameters:
    ===========
    input_tsv: Input tsv file
    smiles_col: Name of the Smiles column

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    df = pd.read_csv(input_tsv, sep="\t")

    if parse_smiles and smiles_col in df.columns:
        df[smiles_col] = df[smiles_col].map(smiles_to_mol)

    return df


def write_tsv(df: pd.DataFrame, output_tsv: str, smiles_col: str = "Smiles"):
    """Write a tsv file, converting the RDKit molecule column to smiles.
    If the Smiles column contains RDKit Molecules instead of strings, these are converted to Smiles with default parameters.

    Parameters:
    ===========
    input_tsv: Input tsv file
    smiles_col: Name of the Smiles column

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    if len(df) > 0 and smiles_col in df.columns:
        probed = df.iloc[0][smiles_col]
        if isinstance(probed, Mol):
            df[smiles_col] = df[smiles_col].map(mol_to_smiles)
    df.to_csv(output_tsv, sep="\t", index=False)


def count_violations_lipinski(
    molecular_weight: float, slogp: float, num_hbd: int, num_hba: int
) -> int:
    """Apply the filters described in reference (Lipinski's rule of 5) and count how many rules
    are violated. If 0, then the compound is strictly drug-like according to Lipinski et al.

    Ref: Lipinski, J Pharmacol Toxicol Methods. 2000 Jul-Aug;44(1):235-49.

    Parameters:
    ===========
    molecular_weight: Molecular weight
    slogp: LogP computed with RDKit
    num_hbd: Number of Hydrogen Donors
    num_hba: Number of Hydrogen Acceptors

    Returns:
    ========
    The number of violations of the Lipinski's rule.
    """
    n = 0
    if molecular_weight < 150 or molecular_weight > 500:
        n += 1
    if slogp > 5:
        n += 1
    if num_hbd > 5:
        n += 1
    if num_hba > 10:
        n += 1
    return n


def count_violations_veber(num_rotatable_bonds: int, tpsa: float) -> int:
    """Apply the filters described in reference (Veber's rule) and count how many rules
    are violated. If 0, then the compound is strictly drug-like according to Veber et al.

    Ref: Veber DF, Johnson SR, Cheng HY, Smith BR, Ward KW, Kopple KD (June 2002).
    "Molecular properties that influence the oral bioavailability of drug candidates".
    J. Med. Chem. 45 (12): 2615–23.

    Parameters:
    ===========
    num_rotatable_bonds: Number of rotatable bonds
    tpsa: Topological Polar Surface Area

    Returns:
    ========
    The number of violations of the Veber's rule.
    """
    n = 0
    if num_rotatable_bonds > 10:
        n += 1
    if tpsa > 140:
        n += 1
    return n


def get_min_ring_size(mol: Mol) -> int:
    """Return the minimum ring size of a molecule. If the molecule is linear, 0 is returned.

    Parameters:
    ===========
    mol: The input molecule

    Returns:
    ========
    The minimal ring size of the input molecule
    """

    ring_sizes = [len(x) for x in mol.GetRingInfo().AtomRings()]
    try:
        return min(ring_sizes)
    except ValueError:
        return 0


def get_max_ring_size(mol: Mol) -> int:
    """Return the maximum ring size of a molecule. If the molecule is linear, 0 is returned.

    Parameters:
    ===========
    mol: The input molecule

    Returns:
    ========
    The maximal ring size of the input molecule
    """
    ring_sizes = [len(x) for x in mol.GetRingInfo().AtomRings()]
    try:
        return max(ring_sizes)
    except ValueError:
        return 0


def compute_descriptors(mol: Mol, descriptors_list: list = None) -> dict:
    """Compute predefined descriptors for a molecule.
    If the parsing of a molecule fails, then an Nan values are generated for all properties.

    Parameters:
    ===========
    mol: The input molecule
    descriptors_list: A list of descriptors, in case the user wants to compute less than default.

    Returns:
    ========
    A dictionary with computed descriptors with syntax such as descriptor_name: value.
    """
    # predefined descriptors
    descriptors = DESCRIPTORS

    # update the list of descriptors to compute with whatever descriptor names are in the provided list,
    # if the list contains an unknown descriptor, a KeyError will be raised.
    if descriptors_list is not None:
        descriptors = {k: v for k, v in descriptors.items() if k in descriptors_list}

    # parse smiles on the fly
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    # if parsing fails, return dict with missing values
    if mol is None:
        return {k: None for k in descriptors.keys()}

    # run
    try:
        # compute molecular descriptors
        d = {k: v(mol) for k, v in descriptors.items()}
        # annotate subsets
        d["num_violations_lipinski"] = count_violations_lipinski(
            d["molecular_weight"], d["slogp"], d["num_hbd"], d["num_hba"]
        )
        d["num_violations_veber"] = count_violations_veber(
            d["num_rotatable_bond"], d["tpsa"]
        )
    except ValueError:
        d = {k: None for k in descriptors.keys()}

    return d


def compute_descriptors_df(df: DataFrame, smiles_col: str) -> DataFrame:
    """Compute descriptors on the smiles column of a DataFrame.
    The Smiles is parsed on the fly only once.


    Parameters:
    ===========
    df: The input DataFrame
    smiles_col: The name of the column with the molecules in smiles format

    Returns:
    ========
    The input dictionary concatenated with the computed descriptors

    """
    return pd.concat(
        [
            df,
            df.apply(lambda x: compute_descriptors(x[smiles_col]), axis=1).apply(
                pd.Series
            ),
        ],
        axis=1,
    )


def lp(obj, label: str = None, lpad=50, rpad=10):
    """log-printing for different kind of objects"""
    if isinstance(obj, str):
        if label is None:
            label = "String"
        print(f"{label:{lpad}s}: {obj:>{rpad}s}")
        return
    try:
        fval = float(obj)
        if label is None:
            label = "Number"
        if fval == obj:
            print(f"{label:{lpad}s}: {int(obj):{rpad}d}")
        else:
            print(f"{label:{lpad}s}: {obj:{rpad+6}.5f}")
        return
    except (ValueError, TypeError):
        # print("Exception")
        pass

    try:
        shape = obj.shape
        if label is None:
            label = "Shape"
        else:
            label = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.keys())
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        shape = obj.data.shape
        if label is None:
            label = "Shape"
        else:
            label = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.data.keys())
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.data.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        length = len(obj)
        if label is None:
            label = "Length"
        else:
            label = f"len({label})"
        print(f"{label:{lpad}s}: {length:{rpad}d}")
        return
    except (TypeError, AttributeError):
        pass

    if label is None:
        label = "Object"
    print(f"{label:{lpad}s}: {obj}")


def get_pca_feature_contrib(pca_model: PCA, features: list) -> DataFrame:
    """Get the feature contribution to each Principal Component.

    Parameters:
    ===========
    model: The PCA object
    descriptors_list: The list of feature names that were used for the PCA.

    Returns:
    ========
    A DataFrame with the feature contribution.
    """
    # associate features and pc feature contribution
    ds = []
    for pc in pca_model.components_:
        ds.append(
            {k: np.abs(v) for k, v in zip(features, pc)}
        )  # absolute value of contributions because only the magnitude of the contribution is of interest
    df_feature_contrib = (
        pd.DataFrame(ds, index=[f"PC{i+1}_feature_contrib" for i in range(3)])
        .T.reset_index()
        .rename({"index": "Feature"}, axis=1)
    )

    # compute PC ranks
    for c in df_feature_contrib.columns:
        if not c.endswith("_feature_contrib"):
            continue
        df_feature_contrib = df_feature_contrib.sort_values(
            c, ascending=False
        ).reset_index(drop=True)
        df_feature_contrib[f"{c.split('_')[0]}_rank"] = df_feature_contrib.index + 1

    # add PC-wise ratios
    pattern = "_feature_contrib"
    for c in df_feature_contrib:
        if c.endswith(pattern):
            tot = df_feature_contrib[c].sum()
            df_feature_contrib = df_feature_contrib.sort_values(c, ascending=False)
            df_feature_contrib[
                f"{c.replace(pattern, '')}_feature_contrib_cum_ratio"
            ] = (df_feature_contrib[c].cumsum() / tot)

    return df_feature_contrib.sort_values("Feature").reset_index(drop=True)


def format_pca_feature_contrib(df_feature_contrib: DataFrame) -> DataFrame:
    """
    Format a DataFrame with explained variance so that the columns for each PC become
    new rows.

    Parameters:
    ===========
    df_feature_contrib: The DataFrame with PC feature contributions

    Returns:
    ========
    A rearranged DataFrame with the feature contribution, with common column names and each PC as different rows.
    """
    pcs = list(
        set([c.split("_")[0] for c in df_feature_contrib.columns if c.startswith("PC")])
    )
    # init empty DataFrame
    df = pd.DataFrame(None, columns=["PC", "Feature", "Contribution", "Rank"])
    for pc in pcs:
        df_tmp = df_feature_contrib[
            ["Feature", f"{pc}_feature_contrib", f"{pc}_rank"]
        ].rename(
            {f"{pc}_feature_contrib": "Contribution", f"{pc}_rank": "Rank"}, axis=1
        )
        df_tmp["PC"] = pc
        df = pd.concat([df, df_tmp])

    return df.reset_index(drop=True).sort_values(["PC", "Feature"])


def get_pca_var(pca_model: PCA) -> DataFrame:
    """
    Extract the explained variance from a PCA model as a DataFrame.

    Parameters:
    ===========
    pca_model: a PCA model containing the explained variance for each PC.

    Returns:
    ========
    A DataFrame with the explained variance for each PC.
    """
    feature_contrib = pca_model.explained_variance_ratio_
    pcs = [f"PC{i+1}" for i in range(len(feature_contrib))]
    # generate the variance data
    df_pca_var = pd.DataFrame(
        {
            "var": feature_contrib,
            "PC": pcs,
        }
    )
    df_pca_var["var_perc"] = df_pca_var["var"].map(lambda x: f"{x:.2%}")
    df_pca_var["var_cum_ratio"] = df_pca_var["var"].cumsum()  # total is 1
    df_pca_var["var_cum_perc"] = df_pca_var["var_cum_ratio"].map(lambda x: f"{x:.2%}")

    return df_pca_var


def plot_pca_var(df_pca_var: DataFrame) -> plt.Figure:
    """Plot the explained variance of each Principal Component.

    Parameters:
    ===========
    df_pca_var: a DataFrame with the PCA explained variance

    Returns:
    ========
    A barplot with the explained variance for each Principal Component.
    """
    total_var = df_pca_var["var"].sum()
    df_pca_var["n"] = df_pca_var["PC"].map(lambda x: int(x[2:]))

    # generate the variance plot
    # initplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 9))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    # fig.suptitle("Variance Explained by Principal Components", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    x_label = "Number of Principal Components"
    y_label = "Cumulated % of Total Variance"
    # create a red dotted line at 60%
    ax.axhline(0.6, ls="--", color="red", zorder=1)
    # create the bar plot
    sns.barplot(
        ax=ax, x="n", y="var_cum_ratio", data=df_pca_var, color="gray", zorder=2
    )

    # customize the plot
    ax.set_title("Variance Explained by Principal Components", fontsize=30, y=1.02)
    ax.tick_params(labelsize=20)
    ax.set_xlabel(x_label, fontsize=25, labelpad=20)
    ax.set_ylabel(y_label, fontsize=25, labelpad=20)
    ylabels = [f"{x:,.0%}" for x in ax.get_yticks()]
    ax.set_yticklabels(ylabels)

    # add % on the bars
    for a, i in zip(ax.patches, range(len(df_pca_var.index))):
        row = df_pca_var.iloc[i]
        ax.text(
            row.name,
            a.get_height(),
            row["var_cum_perc"],
            color="black",
            ha="center",
            fontdict={"fontsize": 20},
        )

    plt.tight_layout()
    figure = ax.get_figure()
    return figure


def plot_pc_proj(
    df_pca: DataFrame,
    palette,
    hue_order=["ChEMBL-NP", "DrugBank", "Enamine", "Pseudo-NPs"],
) -> plt.Figure:
    """Plot the PCA data projected into the PC space.

    Parameters:
    ===========
    df_pca: a DataFrame with the PCA data
    hue_order: The order in which to plot the datasets
    palette: a list of colors to use for the datasets

    Returns:
    ========
    A multi scatterplot, with a subplot for each Principal Component Combination.
    """
    # sort df by the hue order
    df_pca = df_pca.copy()
    df_pca["Dataset"] = df_pca["Dataset"].astype("category")
    df_pca["Dataset"].cat.set_categories(hue_order, inplace=True)
    df_pca = df_pca.sort_values(["Dataset"])

    # initiate the multiplot
    fig_size = (32, 12)
    sns.set(rc={"figure.figsize": fig_size})
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.suptitle("Principal Component Analysis", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)

    # iterate over the possible combinations
    for i, col_pairs in enumerate([["PC1", "PC2"], ["PC1", "PC3"], ["PC2", "PC3"]]):
        plt.subplot(1, 3, i + 1)
        x_label = col_pairs[0]
        y_label = col_pairs[1]
        ax = sns.scatterplot(
            x=x_label,
            y=y_label,
            data=df_pca,
            hue="Dataset",  # color by cluster
            legend=True,
            palette=palette,
            alpha=0.5,
            edgecolor="none",
        )
        ax.set_title(f"{x_label} and {y_label}", fontsize=24, y=1.02)

    # clean up plot
    plt.tight_layout()
    figure = ax.get_figure()
    figure.subplots_adjust(bottom=0.2)
    return figure


def plot_pc_proj_with_ref(
    df_pca: DataFrame, ref: str, palette: Union[str, List] = PALETTE
) -> plt.Figure:
    """Plot the PCA data projected into the PC space.
    A new row is added to the multiplot for each combination reference - subset.

    Parameters:
    ===========
    df_pca: a DataFrame with the PCA data
    ref: the label of the dataset to use as reference
    palette: a list of colors to use for the datasets

    Returns:
    ========
    A multi scatterplot, with a subplot for each combination of ref - dataset and each Principal Component Combination.
    """
    fig_size = (32, 32)

    sns.set(rc={"figure.figsize": fig_size})
    fig = plt.figure()
    fig.subplots_adjust(hspace=1, wspace=0.2)
    fig.suptitle("Principal Component Analysis", fontsize=40, y=1)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    dataset_pairs = itertools.product(
        *[[ref], [e for e in df_pca["Dataset"].unique() if e != ref]]
    )
    counter = 1
    for i, dataset_pair in enumerate(dataset_pairs):

        for col_pairs in [["PC1", "PC2"], ["PC1", "PC3"], ["PC2", "PC3"]]:
            plt.subplot(3, 3, counter)
            x_label = col_pairs[0]
            y_label = col_pairs[1]
            palette_curr = [palette[i + 1], palette[0]]
            ax = sns.scatterplot(
                x=x_label,
                y=y_label,
                data=df_pca[df_pca["Dataset"].isin(dataset_pair)].iloc[
                    ::-1
                ],  # reverse order to get the ref above the rest
                hue="Dataset",  # color by cluster
                legend=True,
                palette=palette_curr,
                alpha=0.5,
                edgecolor="none",
            )
            ax.set_ylim([-10, 15])
            ax.set_xlim([-10, 25])
            ax.set_title(f"{x_label} and {y_label}", fontsize=30, y=1.02)
            counter += 1

    plt.tight_layout(pad=4.8)
    figure = ax.get_figure()

    figure.subplots_adjust(bottom=1.0, top=2.5)
    plt.tight_layout()
    return figure


def plot_pca_cum_feature_contrib_3pc(df_pca_feature_contrib: DataFrame) -> plt.Figure:
    """Plot the cumulated feature contribution to each Principal Component Combination
    individually (up to 3 different PCs).

    Parameters:
    ===========
    df_pca_feature_contrib: a DataFrame with the feature contributions

    Returns:
    ========
    A multi barchart, with a subplot for each Principal Component Combination.
    """
    # set up a multiplot for 3 subplots on a same row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 12), sharey=True)

    # configure plot
    fig.suptitle("Cumulated Feature Contribution to Principal Components", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.8)
    x_label = "Features"
    y_label = "Cumulated % of Feature Contribution"

    # iterate over combinations of PCs (PC1 and PC2, PC1 and PC3 and PC2 and PC3)
    pcs = sorted(
        list(set([c.split("_")[0] for c in df_pca_feature_contrib.columns if "_" in c]))
    )
    for i, pc in enumerate(pcs):
        # create the a subplot
        axes[i].set_title(pc, fontsize=30)
        col_y = f"{pc}_feature_contrib_cum_ratio"
        sns.barplot(
            ax=axes[i],
            x="Feature",
            y=col_y,
            data=df_pca_feature_contrib.sort_values(col_y),
            color="gray",
            zorder=2,
        )
        # add x label and ticks for first plot only
        if i == 0:
            # y label
            axes[i].set_ylabel(y_label, fontsize=25, labelpad=20)
            # y ticklabels
            yticklabels = [f"{x:,.0%}" for x in axes[i].get_yticks()]
            axes[i].set_yticklabels(yticklabels)
        else:
            axes[i].set_ylabel("")

        # x label
        axes[i].set_xlabel(x_label, fontsize=20, labelpad=20)
        axes[i].tick_params(axis="x", rotation=90)
        axes[i].axhline(0.5, ls="--", color="red", zorder=1)

    fig.subplots_adjust(bottom=1.0, top=2.5)
    plt.tight_layout()
    return fig


def plot_pca_feature_contrib(df_pca_feature_contrib: DataFrame) -> plt.Figure:
    """Plot the feature contribution to each Principal Component individually.

    Parameters:
    ===========
    df_pca_feature_contrib: a DataFrame with the feature contributions

    Returns:
    ========
    A multi barchart, with a subplot for each Principal Component.
    """
    fig_size = (32, 12)
    sns.set(rc={"figure.figsize": fig_size})
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)

    g = sns.catplot(
        data=df_pca_feature_contrib,
        kind="bar",
        x="Feature",
        y="Contribution",
        hue="PC",
        ci="sd",
        palette="gray",
        size=18,
        legend=False,
    )
    for ax in g.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    g.fig.suptitle(
        "Feature Contribution to Principal Components",
        fontsize=30,
    )
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    fig = plt.gcf()

    return fig


def plot_pca_loadings_3pc(
    pca_model: PCA, pca: np.ndarray, features: List[str], color_dots: str = "white"
):
    """Plot the principal component loadings. By default, only the arrows
    and the feature labels are plotted. If the color_dots parameter is modified,
    then a biplot is generated instead.

    Parameters:
    ===========
    pca_model: the PCA model
    pca: the PCA data
    descriptors_list: the list of feature names that were used for the PCA.

    Returns:
    ========
    A PCA loadings plot or a PCA biplot.
    """

    # data initialization
    pcs = [f"PC{i}" for i in range(1, pca.shape[1] + 1)]
    pc_pairs = list(itertools.combinations(pcs, 2))
    scores = pca
    coefficients = np.transpose(pca_model.components_)

    # plot initialization
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(24, 7), sharex=True, sharey=True
    )
    axes = axes.ravel()
    # add main title
    fig.suptitle("Principal Component Loading", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.8)

    # generate one subplot at the time
    for i, ax in enumerate(axes):
        pc_pair = pc_pairs[i]
        # determine what columns to retrieve from the pca matrix
        idx_x = int(pc_pair[0].replace("PC", "")) - 1
        idx_y = int(pc_pair[1].replace("PC", "")) - 1
        # retrieve values
        scores_x = scores[:, idx_x]
        scores_y = scores[:, idx_y]
        coefficients_curr = coefficients[:, [idx_x, idx_y]]
        # zoom in Principal Components space
        n = coefficients_curr.shape[0]
        scale_x = 1.0 / (scores_x.max() - scores_x.min())
        scale_y = 1.0 / (scores_y.max() - scores_y.min())
        # plot all data points as white just to get the appropriate coordinates
        ax.scatter(
            x=scores_x * scale_x, y=scores_y * scale_y, s=5, color="white"
        )  # we interest ourselves in the loadings, this is not a biplot
        # add eigenvectors as annotated arrows
        for j in range(n):
            ax.arrow(
                0,
                0,
                coefficients_curr[j, 0],
                coefficients_curr[j, 1],
                color="gray",
                alpha=0.8,
                head_width=0.015,
            )
            ax.text(
                coefficients_curr[j, 0],
                coefficients_curr[j, 1],
                features[j],
                color="red",
                ha="center",
                va="center",
                fontsize=11,
            )

        # finish subplots
        ax.set_title(f"{' and '.join(pc_pair)}", fontsize=20)
        ax.set_xlim([-0.8, 0.8])
        ax.set_xlabel(pc_pair[0], fontsize=20, labelpad=20)
        ax.set_ylabel(pc_pair[1], fontsize=20, labelpad=20)

    return fig


# FRAGMENTS = {
#    "acyl_halide": Chem.MolFromSmarts('[#9,#17,#35,#53]=O'),  # C(=O)X
#    "anhydride": Chem.MolFromSmarts('[#6]-[#6](=O)-[#8]-[#6](-[#6])=O'),  # CC(=O)OC(=O)C
#    "peroxide": Chem.MolFromSmarts('[#8]-[#8]'),  # R-O-O-R'
#    "ab_unsaturated_ketone": Chem.MolFromSmarts('[#6]=[#6]-[#6]=O'),  # R=CC=O
# }

DESCRIPTORS = {
    # classical molecular descriptors
    "num_heavy_atoms": lambda x: x.GetNumAtoms(),
    "molecular_weight": lambda x: round(Desc.ExactMolWt(x), 4),
    "num_rings": lambda x: rdMolDesc.CalcNumRings(x),
    "num_rings_arom": lambda x: rdMolDesc.CalcNumAromaticRings(x),
    "num_rings_ali": lambda x: rdMolDesc.CalcNumAliphaticRings(x),
    "num_hbd": lambda x: rdMolDesc.CalcNumLipinskiHBD(x),
    "num_hba": lambda x: rdMolDesc.CalcNumLipinskiHBA(x),
    "slogp": lambda x: round(Crippen.MolLogP(x), 4),
    "tpsa": lambda x: round(rdMolDesc.CalcTPSA(x), 4),
    "num_rotatable_bond": lambda x: rdMolDesc.CalcNumRotatableBonds(x),
    "num_atoms_oxygen": lambda x: len(
        [a for a in x.GetAtoms() if a.GetAtomicNum() == 8]
    ),
    "num_atoms_nitrogen": lambda x: len(
        [a for a in x.GetAtoms() if a.GetAtomicNum() == 7]
    ),
    "num_atoms_halogen": Fragments.fr_halogen,
    "num_atoms_bridgehead": rdMolDesc.CalcNumBridgeheadAtoms,
    # custom molecular descriptors
    # "ring_size_min": get_min_ring_size,
    # "ring_size_max": get_max_ring_size,
    "frac_sp3": lambda x: rdMolDesc.CalcFractionCSP3(x),
    # HTS filters 1/2 - present in the RDKit Fragments
    # "num_aldehyde": Fragments.fr_aldehyde,
    # "num_diazo":Fragments.fr_diazo,
    # "num_carbonyl": Fragments.fr_C_O,  # in Over paper, dicarbonyl compounds are filtered out
    # "num_sulfide": Fragments.fr_sulfide,  # in Over paper, disulfide compounds are filtered out
    # "num_hydrazine": Fragments.fr_hdrzine,
    # "num_isocyanate": Fragments.fr_isocyan,
    # "num_isothiocyanate": Fragments.fr_isothiocyan,
    # "num_quaternary_amine": Fragments.fr_quatN,
    # HTS filters 2/2 - not present in the RDKit Fragments
    # "num_ab_unsaturated_ketone": lambda x: len(x.GetSubstructMatches(FRAGMENTS['ab_unsaturated_ketone'])),  # R=CC=O
    # "num_acyl_halide": lambda x: len(x.GetSubstructMatches(FRAGMENTS['acyl_halide'])),  # C(=O)X
    # "num_anhydride": lambda x: len(x.GetSubstructMatches(FRAGMENTS['anhydride'])),  # CC(=O)OC(=O)C
    # "num_peroxide": lambda x: len(x.GetSubstructMatches(FRAGMENTS['peroxide'])),  # R-O-O-R'
}


def plot_features(df_features: DataFrame, dataset_name: str, color: str) -> plt.Figure:
    """Create a multiplot of maximum 5x4 sub-barplots, with one barplot for each feature.

    Note: superfluous subplots (in case there are less than 20 features) are not plotted.

    Parameters:
    ===========
    df_features: The DataFrame with computed features
    dataset_name: The name of the dataset to print in the suptitle
    color: the color to use for the bars

    Returns:
    ========
    A figure with 5x4 subplots.
    """
    # count the number of computed features
    features = [
        c
        for c in df_features.columns
        if c
        in list(DESCRIPTORS.keys())
        + ["num_violations_lipinski", "num_violations_veber"]
    ]
    num_features = len(features)

    # init figure
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(24, 20))
    axes = axes.ravel()  # access directly the ax objects

    # add main title
    fig.suptitle(f"Feature Distribution in {dataset_name}", fontsize=40)  # , y=0.92)
    # set style
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # , top=0.8)

    # plot barcharts
    for i, ax in enumerate(axes):

        if i < num_features:
            sns.distplot(
                df_features[features[i]],
                kde=False,
                label="",
                color=color,
                ax=ax,
                hist_kws=dict(alpha=1),
            )
            feature_mean = df_features[features[i]].mean()
            ax.axvline(
                feature_mean, color="black", ls="--", zorder=2
            )  # dotted line for median

            if not i % 5:
                ax.set_ylabel("Count", fontsize=20)  # , labelpad=20)

            # ax.set_title(f"{features[i]}", fontsize=20)
        else:
            # here need to hide mol descriptors for empty plots (18th, 19th, 20th)
            fig.delaxes(ax)
    plt.tight_layout()

    return fig
