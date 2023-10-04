import pandas as pd
import re


def filter_mo_analysis(dataframe: pd.DataFrame, index: list = [], mo_list=None, atoms: dict = None):
    columns = [i for i in list(dataframe) if 'sum' in i or 'MO' in i]
    index_list = list(dataframe.index.names)
    if atoms:
        dataframe = dataframe.reset_index()
        for label, atom_list in atoms.items():
            if isinstance(atom_list, str):
                atom_list = range_translator(atom_list)
            dataframe.loc[dataframe["Atom Number"].isin(atom_list), "Atom Type"] = label
        dataframe.set_index(index_list, inplace=True)
    if index:
        if "Info" in index_list:
            index = ["Info"] + index
        dataframe = dataframe.groupby(index, sort=False)[columns].sum()
    if mo_list:
        if isinstance(mo_list, str):
            mo_list = range_translator(mo_list)
        kept_mo = ['MO ' + str(number) for number in mo_list]
        dataframe = dataframe[kept_mo]
    return dataframe

def mo_analysis_transpose(dataframe):
    df = pd.concat([filter_mo_analysis(dataframe, ["Atom Type"]),
                    filter_mo_analysis(dataframe, ["Atom Type", "Angular Momentum"]).reset_index(
                        level="Angular Momentum").iloc[2:]], axis=0).reset_index(level="Atom Type")
    df['ml'] = df['Angular Momentum'].map({'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6})
    df = df.assign(group=pd.factorize(df['Atom Type'])[0]).sort_values(['group', 'ml'], na_position='first').drop(
        columns=['group', 'ml'])
    df['Angular Momentum'] = df['Angular Momentum'].fillna(value="")
    df.set_index(['Atom Type', 'Angular Momentum'], append=True, inplace=True)
    return df.transpose()


def filter_transition_rows(dataframe: pd.DataFrame, index: list = [], attributes: list = ["transition energy", "oscillator strength"], energy_range: tuple = None, atoms: dict = None):
    if energy_range is not None:
        dataframe = dataframe[(dataframe['transition energy'] >= energy_range[0]) & (dataframe['transition energy'] <= energy_range[1])]
    columns = [i for i in list(dataframe) if 'sum' in i or 'MO' in i]
    if atoms:
        dataframe = dataframe.reset_index(level=["Atom Number", "Atom Type",
                             "Shell Number", "Angular Momentum",
                             "Magnetic QN"])
        for label, atom_list in atoms.items():
            if isinstance(atom_list, str):
                atom_list = range_translator(atom_list)
            dataframe.loc[dataframe["Atom Number"].isin(atom_list), "Atom Type"] = label
        dataframe.set_index(["Atom Number", "Atom Type",
                             "Shell Number", "Angular Momentum",
                             "Magnetic QN"], append=True)
    group_list = ["Starting State", "Ending State"] + index + attributes
    dataframe = dataframe.groupby(group_list, sort=False)[columns].sum()
    dataframe.reset_index(level=attributes, inplace=True)
    return dataframe


def range_translator(string):
    range_list = []
    string = re.sub(r"[a-zA-Z()]+", "", string).replace("[", "").replace("]", "").replace(",", " ").split()
    for current_string in string:
        if ":" in current_string:
            range_val = current_string.split(":")
            range_list += list(range(int(range_val[0]), int(range_val[1]) + 1))
        else:
            range_list.append(int(current_string))
    return range_list


def filter_transition_columns(dataframe, custom_mo_dict, box):
    new_df = dataframe.copy()
    mo_dict = {}
    if isinstance(custom_mo_dict, str):
        custom_mo_dict = range_translator(custom_mo_dict)
    if isinstance(custom_mo_dict, list):
        mo_list = custom_mo_dict
        custom_mo_dict = {}
    elif isinstance(custom_mo_dict, dict):
        mo_list = []
        for label, mo in custom_mo_dict.items():
            if isinstance(mo, str):
                mo = range_translator(mo)
            mo_list += mo
            custom_mo_dict[label] = ['AS MO ' + str(number) for number in mo]
    start = "AS MO " + str(box.basic_data.n_mo) in dataframe.columns
    core_column = ['AS MO ' + str(mo) for mo in range(start, box.basic_data.homo + start) if mo not in mo_list]
    mo_dict["core"] = core_column
    mo_dict.update(custom_mo_dict)
    valence_column = ['AS MO ' + str(mo) for mo in range(box.basic_data.homo + start, box.basic_data.n_mo + start) if mo not in mo_list]
    mo_dict["valence"] = valence_column
    for label, column_list in mo_dict.items():
        combine_mo_columns(new_df, label, column_list)
    return new_df


def show_nonzero_mo(dataframe):
    new_df = dataframe.copy()
    if "core" in dataframe.columns:
        columns = ['core', 'valence']
    else:
        columns = [i for i in list(dataframe) if 'MO' in i]
    column_index = new_df.columns.get_indexer([columns[0], columns[-1]])
    cols = new_df.columns.values[column_index[0]:]
    mask = new_df.ne(0).values[:, column_index[0]:]
    out = [cols[x].tolist() for x in mask]
    new_df.insert(new_df.columns.get_loc(columns[0]), "non-zero MOs", out)
    return new_df


def get_nonzero_mo(dataframe):
    non_zero_mo_per_excitation = dataframe["non-zero MOs"].to_list()
    mo_list = set()
    for current_mo_list in non_zero_mo_per_excitation:
        mo_list.update(current_mo_list)
    mo_list = [int(mo.strip("AS MO")) for mo in mo_list]
    mo_list.sort()
    return mo_list


def combine_mo_columns(dataframe, label, column_list):
    sum_column = dataframe.loc[:, column_list].sum(axis=1)
    if label == "core":
        dataframe.insert(dataframe.columns.get_loc("AS MO 1"), label, sum_column)
    else:
        dataframe[label] = sum_column
    dataframe.drop(column_list, axis=1, inplace=True)
