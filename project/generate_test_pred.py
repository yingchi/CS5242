import numpy as np
import pandas as pd
from math import floor
from keras.models import load_model
from read_testing_pdb_file import read_pdb
from train_cnn_model import point_within_vox, convert_xyz_to_vox


def create_data_rec(pro, lig, label,
                    pro_list, lig_list, label_list, width, unit):
    file_name = "{}_{}_cg.pdb".format(("0000" + str(lig))[-4:], "lig")
    X_list, Y_list, Z_list, atomtype_list = read_pdb("./testing_data/{}".format(file_name))
    x0 = floor(np.mean(X_list))
    y0 = floor(np.mean(Y_list))
    z0 = floor(np.mean(Z_list))
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    lig_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    file_name = "{}_{}_cg.pdb".format(("0000" + str(pro))[-4:], "pro")
    X_list, Y_list, Z_list, atomtype_list = read_pdb("./testing_data/{}".format(file_name))
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    pro_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    label_list.append(label)


def get_pred_df_for_pro(pro, lig_list, model, width, unit):
    pro_test, lig_test, label_test = [], [], []
    for lig in lig_list:
        create_data_rec(pro, lig, -1, pro_test, lig_test, label_test, width, unit)
    testset = np.concatenate([pro_test, lig_test], axis=4)
    pred = model.predict(testset).flatten()
    d = {'idx': lig_list, 'prob': list(pred.flatten())}
    pred_top20 = pd.DataFrame(d).sort_values(by='prob', ascending=False).head(20).set_index('idx')
    pred_top20['score'] = list(range(20, 0, -1))

    return pred_top20


if __name__ == '__main__':
    # ============================================
    # PLEASE CHANGE THIS PART FOR DIFFERENT MODELS
    width1 = 10
    unit1 = 1
    trained_model_1 = load_model('./output_model/m2800_w10.03-0.1113-0.9570-0.1258-0.9536.hdf5')

    width2 = 20
    unit2 = 2
    trained_model_2 = load_model('./output_model/m2800_w20.05-0.0938-0.9649-0.1397-0.9452.hdf5')

    width3 = 30
    unit3 = 3
    trained_model_3 = load_model('./output_model/m2800_w30.05-0.1453-0.9463-0.1766-0.9345.hdf5')

    num_test = 825
    # ============================================

    top10_list = []
    lig_list = list(range(1, num_test))

    for pro in range(1, num_test):
        if pro % 50 == 0:
            print("Trying to match for pro_{}".format(pro))
        pred1_top20 = get_pred_df_for_pro(pro, lig_list, trained_model_1, width1, unit1)
        pred2_top20 = get_pred_df_for_pro(pro, lig_list, trained_model_2, width2, unit2)
        pred3_top20 = get_pred_df_for_pro(pro, lig_list, trained_model_3, width3, unit3)

        pred_top20 = pred1_top20.add(pred2_top20, fill_value=0).add(pred3_top20, fill_value=0)
        top10_idx = pred_top20.sort_values(by=['prob', 'score'],
                                           ascending=[False, False]).head(10).index
        print(pro, list(top10_idx))
        top10_list.append([pro] + list(top10_idx))

    header = ['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id',
              'lig6_id', 'lig7_id', 'lig8_id', 'lig9_id', 'lig10_id']
    out = pd.DataFrame(top10_list, columns=header)
    out.to_csv('test_predictions.txt', index=False)
