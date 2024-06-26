from utils import load_class_splits
import random 
import os
from torch.utils.data import Dataset
from PIL import Image

class CUBCustomDataset(Dataset):
    def __init__(self, image_paths, transform, label_map):
        self.image_paths = image_paths
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert('RGB')

        label = self.label_map[int(image_filepath.split('/')[-2].split('.')[0]) - 1]
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print(image_filepath)

        return image, label
    
def get_cub_splits():
    random.seed(10)
    known = load_class_splits('cub')['known_classes']
    easy_unknown = load_class_splits('cub')['unknown_classes']['Easy']
    medium_unknown = load_class_splits('cub')['unknown_classes']['Medium']
    hard_unknown = load_class_splits('cub')['unknown_classes']['Hard']

    ins = []
    easy_oos = []
    med_oos = []
    hard_oos = []

    dr = '/home/avisund/data/CUB/CUB_200_2011/images/'
    folder_names = os.listdir(dr)

    for i in range(len(folder_names)):
        if '.DS_Store' in folder_names[i]:
            continue
        if i in known:
            ins.append(folder_names[i])
        elif i in easy_unknown:
            easy_oos.append(folder_names[i])
        elif i in medium_unknown:
            med_oos.append(folder_names[i])
        elif i in hard_unknown:
            hard_oos.append(folder_names[i])

    train_list = []
    test_list = []
    train_labels = []
    test_labels = []
    label_count = 0

    label_map = {}

    for i in range(len(ins)):
        imgs = os.listdir(dr + ins[i])
        test = random.sample(imgs, int(0.2 * len(imgs)))
        train_list += [dr + ins[i] + '/' + elem for elem in imgs if elem not in test]

        label = int(ins[i].split('.')[0]) - 1
        train_labels += (len(imgs) - int(0.2 * len(imgs))) * [label_count]
        test_list += [dr + ins[i] + '/' + elem for elem in test]
        test_labels += int(0.2 * len(imgs)) * [label_count]
        label_map[label] = label_count
        label_count += 1

    easy_oos_list = []
    med_oos_list = []
    hard_oos_list = []
    easy_oos_labels = []
    med_oos_labels = []
    hard_oos_labels = []
    for elem in easy_oos:
        easy_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
        label = int(elem.split('.')[0]) - 1
        easy_oos_labels += [label_count] * len(os.listdir(dr + elem))
        label_map[label] = label_count
        label_count += 1

    for elem in med_oos:
        med_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
        label = int(elem.split('.')[0]) - 1
        med_oos_labels += [label_count] * len(os.listdir(dr + elem))
        label_map[label] = label_count
        label_count += 1

    for elem in hard_oos:
        hard_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
        label = int(elem.split('.')[0]) - 1
        hard_oos_labels += [label_count] * len(os.listdir(dr + elem))
        label_map[label] = label_count
        label_count += 1

    return train_list, test_list, easy_oos_list, med_oos_list, hard_oos_list, label_map


