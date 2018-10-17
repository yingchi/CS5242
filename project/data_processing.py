import numpy as np
import pandas as pd
from math import floor
from read_pdb_file import read_pdb

## raw data files read in function
def read_raw_data(n, file_type):
    if (n < 1) or (n > 3000):
        return "Please provide a valid number of files between 1 to 3000"
    if (file_type != "lig") and (file_type != "pro"):
        return "Please provide a valid file type which should be 'lig' or 'pro'"
    res = []
    for i in range(1, n+1):
        file_number = ("0000"+str(i))[-4:]
        file_name = "{}_{}_cg.pdb".format(file_number, file_type)
        X_list, Y_list, Z_list, atomtype_list=read_pdb("./training_data/{}".format(file_name))
        res.append([X_list, Y_list, Z_list, atomtype_list])
    return res

## read in protein and ligand data
meta_pro = read_raw_data(10, "pro")
meta_lig = read_raw_data(10, "lig")

## create training data
data = [meta_lig[0]]
print(data)
min_x, max_x, unit_len_x = 30, 40, 1
min_y, max_y, unit_len_y = 32, 38, 1
min_z, max_z, unit_len_z = 18, 28, 1
df = pd.DataFrame(columns=["dim_x_h", "dim_x_p",
                           "dim_y_h", "dim_y_p",
                           "dim_z_h", "dim_z_p"])
for r in range(0, len(data)):
    # cube initialization
    dim_x_h = np.array([0]*(int((max_x-min_x)/unit_len_x)))
    dim_x_p = np.array([0]*(int((max_x-min_x)/unit_len_x)))
    dim_y_h = np.array([0]*(int((max_y-min_y)/unit_len_y)))
    dim_y_p = np.array([0]*(int((max_y-min_y)/unit_len_y)))
    dim_z_h = np.array([0]*(int((max_z-min_z)/unit_len_z)))
    dim_z_p = np.array([0]*(int((max_z-min_z)/unit_len_z)))
    # get all atoms info in one record
    atom_x = data[r][0]
    atom_y = data[r][1]
    atom_z = data[r][2]
    atom_type = data[r][3]
    # update info in cube for each atom
    for i in range(0, len(atom_type)):
        # check atom is in range of defined cube
        check_x = (atom_x[i]>=min_x) and (atom_x[i]<=max_x)
        check_y = (atom_y[i]>=min_y) and (atom_y[i]<=max_y)
        check_z = (atom_z[i]>=min_z) and (atom_z[i]<=max_z)
        if check_x and check_y and check_z:
            # calculate index of cube to be updated
            index_x = floor((atom_x[i]-min_x)/unit_len_x)
            index_y = floor((atom_y[i]-min_y)/unit_len_y)
            index_z = floor((atom_z[i]-min_z)/unit_len_z)
            if atom_type[i]=='h':
                dim_x_h[index_x] += 1
                dim_y_h[index_y] += 1
                dim_z_h[index_z] += 1
            elif atom_type[i]=='p':
                dim_x_p[index_x] += 1
                dim_y_p[index_y] += 1
                dim_z_p[index_z] += 1
    df.loc[r+1] = [dim_x_h, dim_x_p, dim_y_h, dim_y_p, dim_z_h, dim_z_p]

df.to_csv("test_res.csv")
