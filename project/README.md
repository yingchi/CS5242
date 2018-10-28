## CS5242 Project 

### Pre-requisite

#### Dependencies
The code is tested with python 3.6. 
Make sure that you have the packages specified in `requirements.txt` installed. If not, 
please run `pip install -r requirements.txt`

#### Data and folder structure
You should have the following folder structures:

```sh
./
├── README.md
├── train_cnn_model.py
├── generate_test_pred.py
├── model_evaluation.py
├── read_pdb_file.py
├── read_testing_pdb_file.py
├── training_data
│   ├── 0001_lig_cg.pdb
│   ├── 0001_pro_cg.pdb
│   ├── ...
│   ├── 3000_lig_cg.pdb
│   └── 3000_pro_cg.pdb
├── testing_data
│   ├── 0001_lig_cg.pdb
│   ├── 0001_pro_cg.pdb
│   ├── ...
│   ├── 0824_lig_cg.pdb
│   └── 0824_pro_cg.pdb
├── output_model
│   ├── m2800_w10.03-0.1113-0.9570-0.1258-0.9536.hdf5
│   ├── m2800_w20.05-0.0938-0.9649-0.1397-0.9452.hdf5
│   └── m2800_w30.05-0.1453-0.9463-0.1766-0.9345.hdf5
└── requirements.txt
```
In `output_model` folder, we have provided 3 trained models. If you prefer to train new models on yourself, just run `train_cnn_model.py` with the parameters you want. Remember to update the model name in the subsequent scripts if you want to run them.

### To run
There are a few scripts that perform different tasks:
1. `train_cnn_model.py`: Train a cnn model based on specified width and unit. You can change it
at the beginning part of the main process. The best model (in terms of val_loss) will be saved during
the training process. 
2. `model_evaluation.py`: Evaluate the trained model using the "test data" (split from the training_data). It will output a confusion matrix for the model predictions. And it will also output the accuracy based on top10 guesses.
3. `generate_test_pred.py`: Generate the top10 predictions for the actual testing_data. It will use a majority vote approach. Thus it requires three trained models before running this script.
