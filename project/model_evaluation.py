import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from train_cnn_model import create_data_rec


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap=None, normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_mat.png')
    # plt.show()


# ============================================
# PLEASE CHANGE THIS PART FOR DIFFERENT MODELS
width = 10
unit = 1
model = load_model('output_model/m2800_w10.03-0.1113-0.9570-0.1258-0.9536.hdf5')
# ============================================


total_idx = [x+1 for x in list(range(3000))]
# set seed so that every model will use the same test set
np.random.seed(7)
test_idx = [x+1 for x in np.random.choice(3000, 200, replace=False)]
train_idx = list(set(total_idx).difference(set(test_idx)))

# ============================
# Predictions with ratio 1:2
# ============================
print('*'*30)
print('Start model predictions and save confusion matrix to plot ...')
pro_test = []
lig_test = []
label_test = []
for idx in test_idx:
    create_data_rec(idx,idx,1,pro_test,lig_test,label_test,width,unit)
    idx_false1 = random.choice(list(set(test_idx) - set([idx])))
    create_data_rec(idx,idx_false1,0,pro_test,lig_test,label_test,width,unit)
    idx_false2 = random.choice(list(set(test_idx) - set([idx, idx_false1])))
    create_data_rec(idx,idx_false2,0,pro_test,lig_test,label_test,width,unit)

pro_test = np.array(pro_test)
lig_test = np.array(lig_test)
X_test = np.concatenate([pro_test, lig_test], axis=4)
y_test = np.array(label_test)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict(X_test)
predictions = [round(x) for x in predictions.reshape(len(y_test),)]
cm = confusion_matrix(predictions, y_test)
plot_confusion_matrix(cm, ['nobind', 'bind'], normalize=False)

# ============================
# Top10 predictions
# ============================
print('*'*30)
print('Start top10 predictions ...')
check_top_10 = []
i = 1
for idx in test_idx:
    if i % 20 == 0:
        print(i, 'done')
    i += 1
    pro_test = []
    lig_test = []
    label_test = []
    for l in test_idx:
        create_data_rec(idx,l,1,pro_test,lig_test,label_test,width,unit)
    testset = np.concatenate([pro_test, lig_test], axis=4)
    pred = model.predict(testset)
    d = {'idx':test_idx,'prob':list(pred.flatten())}
    pred_df = pd.DataFrame(d)
    check = idx in pred_df.sort_values(by='prob', ascending=False).head(10)['idx'].values
    check_top_10.append(check)

total = len(check_top_10)
pred_true = np.sum(check_top_10)
print("Accuracy for top10 predictions:", pred_true/total)
wrong = [i for (i, v) in zip(test_idx, check_top_10) if not v]
print("These proteins do not have correct top10 predictions:", wrong)