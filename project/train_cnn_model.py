import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor

from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from read_pdb_file import read_pdb

DATA_DIR='./training_data/'


def point_within_vox(pt, x_lower=-50, x_upper=50, y_lower=-50,
                     y_upper=50, z_lower=-50, z_upper=50):
    (atom_x, atom_y, atom_z) = pt
    check_x = (atom_x >= x_lower) and (atom_x < x_upper)
    check_y = (atom_y >= y_lower) and (atom_y < y_upper)
    check_z = (atom_z >= z_lower) and (atom_z < z_upper)
    return check_x and check_y and check_z


def convert_xyz_to_vox(atoms_list,
                       x_lower=-50, x_upper=50, y_lower=-50,
                       y_upper=50, z_lower=-50, z_upper=50, unit=1):
    length = int((x_upper-x_lower)/unit)
    vox = np.zeros((length, length, length, 2))

    for atom in atoms_list:
        (x, y, z, t) = atom
        index_x = floor((x-x_lower)/unit)
        index_y = floor((y-y_lower)/unit)
        index_z = floor((z-z_lower)/unit)
        if t =='h':
            vox[index_z, index_y, index_x, 0] += 1
        elif t =='p':
            vox[index_z, index_y, index_x, 1] += 1

    return vox


def create_data_rec(pro, lig, label,
                    pro_list, lig_list, label_list, width, unit):
    file_name = "{}_{}_cg.pdb".format(("0000" + str(lig))[-4:], "lig")
    X_list, Y_list, Z_list, atomtype_list = read_pdb(DATA_DIR + file_name)
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
    X_list, Y_list, Z_list, atomtype_list = read_pdb(DATA_DIR + file_name)
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    pro_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    label_list.append(label)


def cnn_model(length=20):
    input_layer = Input((length, length, length, 4))
    ## convolutional layers
    x = Conv3D(filters=30, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    x = Conv3D(filters=60, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=90, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    ## dense layers
    dense_layer1 = Dense(units=256, activation='relu')(x)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=1, activation='sigmoid')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def check_top10_acc(model):
    check_top_10 = []
    for p in range(1, 501):
        if (p % 20 == 0):
            print("Trying to match for pro_{}".format(p))
        pro_test = []
        lig_test = []
        label_test = []
        for l in range(1, 501):
            create_data_rec(p,l,1,pro_test,lig_test,label_test)
        testset = np.concatenate([pro_test, lig_test], axis=4)
        predres = model.predict(testset)
        check = (p-1) in predres.flatten().argsort()[-10:][::-1]
        check_top_10.append(check)
    return check_top_10


if __name__ == '__main__':
    # ============================================
    # PLEASE CHANGE THIS PART FOR DIFFERENT MODELS
    unit = 1
    width = 10
    length = int(2 * width / unit)
    num_test = 200
    # ============================================

    # ======================================================
    # Random generate 2800 for training and 200 for testing
    # ======================================================
    total_idx = [x+1 for x in list(range(3000))]
    # set seed so that every model will use the same test set
    np.random.seed(7)
    test_idx = [x+1 for x in np.random.choice(3000, num_test, replace=False)]
    train_idx = list(set(total_idx).difference(set(test_idx)))

    pro_train = []
    lig_train = []
    label_train = []

    for idx in train_idx:
        create_data_rec(idx,idx,1,pro_train,lig_train,label_train,width,unit)
        idx_false1 = random.choice(list(set(train_idx) - set([idx])))
        create_data_rec(idx,idx_false1,0,pro_train,lig_train,label_train,width,unit)
        idx_false2 = random.choice(list(set(train_idx) - set([idx, idx_false1])))
        create_data_rec(idx,idx_false2,0,pro_train,lig_train,label_train,width,unit)

    pro_train = np.array(pro_train)
    lig_train = np.array(lig_train)
    X = np.concatenate([pro_train, lig_train], axis=4)
    y = np.array(label_train)

    # ======================================================
    # Training the model
    # ======================================================
    model = cnn_model(length)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])
    model_name = 'm2800_w'+str(width)+\
                 '.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath=model_name, monitor='val_loss', save_best_only=True)]
    losses = model.fit(x=X, y=y, batch_size=10, epochs=30,
                       validation_split=0.2, callbacks = callbacks)

    # ======================================================
    # Plotting for the training & validation losses
    # ======================================================
    train_losses = losses.history['loss']
    val_losses = losses.history['val_loss']

    plt.plot(range(1, len(val_losses)+1), val_losses, 'b.-', label='val_loss')
    plt.plot(range(1, len(train_losses)+1), train_losses, 'g.-', label='train_loss')
    plt.savefig("losses.png")
    # plt.show()
