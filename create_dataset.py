import os

import os.path as osp
import shutil

from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split

PATH_DATAFRAME_FR2 = "/Volumes/ssd/Initflow/dataset-creation/cr_ds/datasetFR2_62/dataframe/dataframe_fr2_62_all.csv"
PATH_DATAFRAME_FR1_DEWARP = "/Volumes/ssd/Initflow/dataset-creation/cr_ds/datasetFR1dewarp/dataframe/dataframe_fr1_dewarp_all.csv"

PATH_IMGS_FR2 = "/Volumes/ssd/DATASETS_GITLAB/FR/invoice-annotation-fr2-fix/imgs_fr2/"
PATH_IMGS_FR1 = "/Volumes/ssd/DATASETS_GITLAB/FR/invoice-annotation-fr1/images/"
PATH_IMGS_FR1_DEWARP = "/Volumes/ssd/DATASETS_GITLAB/FR/invoice-annotation-fr1/dewarp/"

df_fr2 = pd.read_csv(PATH_DATAFRAME_FR2)
df_fr1_dewarp = pd.read_csv(PATH_DATAFRAME_FR1_DEWARP)


def get_split(df, imgs):
    df_files = list(set([os.path.basename(i) for i in imgs]))

    fn2key = dict([(j, os.path.splitext(j)[0]) for i, j in enumerate(df_files)])

    train_ = list(set(df_files))
    train__, test = train_test_split(train_, test_size=0.005, random_state=42, shuffle=True)
    train, val = train_test_split(train__, test_size=0.15, random_state=42, shuffle=True)
    df_train = df[df.file_name.apply(lambda fn: True if fn in train else False)]
    df_val = df[df.file_name.apply(lambda fn: True if fn in val else False)]
    df_test = df[df.file_name.apply(lambda fn: True if fn in test else False)]

    assert set([fn2key[i] for i in df_train.file_name]) & set([fn2key[i] for i in df_val.file_name]) & set([fn2key[i] for i in df_test.file_name]) == set()
    return df_train, df_val, df_test


df_train_fr2, df_val_fr2, df_test_fr2 = get_split(df_fr2, imgs=os.listdir(PATH_IMGS_FR2))
df_train_fr1, df_val_fr1, df_test_fr1 = get_split(df_fr1_dewarp, imgs=os.listdir(PATH_IMGS_FR1) + os.listdir(PATH_IMGS_FR1_DEWARP))

df_train = pd.concat([df_train_fr2, df_train_fr1]).dropna(how='any').reset_index(drop=True)
df_val = pd.concat([df_val_fr2, df_val_fr1]).dropna(how='any').reset_index(drop=True)
df_test = pd.concat([df_test_fr2, df_test_fr1]).dropna(how='any').reset_index(drop=True)

ls = os.getcwd()

os.makedirs(osp.join(ls, "dataset/testing_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/testing_data/annotations"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/val_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/val_data/annotations"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/training_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/training_data/annotations"), exist_ok=True)

shutil.rmtree(osp.join(ls, "dataset"))

os.makedirs(osp.join(ls, "dataset/testing_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/testing_data/annotations"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/val_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/val_data/annotations"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/training_data/images"), exist_ok=True)
os.makedirs(osp.join(ls, "dataset/training_data/annotations"), exist_ok=True)


def create_ds(df, mode="train"):
    if mode == "train":
        data_name = "dataset/training_data"
    elif mode == "test":
        data_name = "dataset/testing_data"
    elif mode == "val":
        data_name = "dataset/val_data"
    else:
        data_name = None

    for file_name, data_ in tqdm(df.groupby(by="file_name")):
        data = data_.reset_index(drop=True)
        if "dewp" in file_name:
            pre = PATH_IMGS_FR1_DEWARP
        elif "fr_2_" in file_name:
            pre = PATH_IMGS_FR2
        else:
            pre = PATH_IMGS_FR1

        shutil.copy(os.path.join(pre, file_name), osp.join(ls, data_name, "images"))

        data.to_csv(osp.join(ls, data_name, "annotations", file_name.replace(".jpg", ".csv")), index=False)


create_ds(df_train, mode="train")
create_ds(df_val, mode="val")
create_ds(df_test, mode="test")

i = None
