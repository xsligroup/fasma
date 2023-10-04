from fasma.core import df_filters as dff


def print_mo_analysis(filename, box, dataframe):
    transposed = dff.mo_analysis_transpose(dataframe)
    transposed.columns
    col_list = list(transposed.columns)
    col_list[1], col_list[0] = col_list[0], col_list[1]
    transposed = transposed[col_list]
    index_list = list(transposed.transpose().index)
    transposed.index.names = ['MO']
    transposed = transposed.reset_index()
    transposed['MO'] = transposed['MO'].str.replace(r'MO', '', regex=True).astype(int)

    with open(filename + ".txt", "w") as external_file:
        header = "NUM_OF_BASIS={}, NUM_OF_ALPHA_ELECTRONS={}, NUM_OF_BETA_ELECTRONS={}".format(box.basic_data.n_basis, box.basic_data.n_alpha_electron, box.basic_data.n_beta_electron)
        print(header, file=external_file)
        mo_str = ""
        mo_str += "\n\n>> MO {:<5} :  {:>8.4f};    Energy: {: f} eV"
        current_atom = None
        for x in range(2, len(index_list)):
            if current_atom is None or current_atom != index_list[x][1]:
                current_atom = index_list[x][1]
                mo_str += "\n   ----  {:>2}".format(current_atom)
            current_l = index_list[x][2]
            if current_l != "":
                mo_str += "    "
            mo_str += str(current_l) + " :   {: .4f};"
        mo_str *= len(transposed)
        print(mo_str.format(*transposed.to_numpy().flatten()), file=external_file)
