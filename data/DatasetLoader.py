import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from itertools import product,permutations
from random import sample, random
from scipy.spatial import distance

#jigsaw_permutations has the top scrumbles parts-> labels = position +1

def calculate_possible_permutations():
    vec = [1,2,3,4,5,6,7,8,9]
    top_30 = [[]]
    top_30_humming = np.zeros(30)
    res = list(permutations(vec))
    for v in res:
        #calculate hamming distance
        dist = distance.hamming(vec, v)
        for i in range(0,30):
            if dist > top_30_humming[i]:
                top_30_humming[i]=dist
                top_30[i]=v
    #now i have top 30 permutations saved 
    global jigsaw_permutations
    jigsaw_permutations = top_30_humming
        
        


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

def prepare_jigsaw_data(names,percent,path):
    samples = len(names)
    amount = int(samples*percent)
    random_index = sample(range(samples),amount)
    name_jigsaw = [names[k] for k in random_index]
    imgs_jigsaw = []
    labels_jigsaw = []
    permutations_jigsaw= []
    #we got the data
    for index in random_index:
        framename = path + '/' + names[index]
        img = Image.open(framename).convert('RGB')
        #transform????
        width, height = img.size
        #i want 3x3 grid
        piece_size = width / 3;
        width_steps = range(0, width, piece_size)
        height_steps = range(0, height, piece_size)
        boxes = ((i, j, i+piece_size, j+piece_size)
                 for i, j in product(width_steps, height_steps))
        parts = [img.crop(box) for box in boxes]
        #lets get one of the permutations
        perms = random.randint(0, 29)
        numbers = jigsaw_permutations[perms];
        permutations_jigsaw.append(numbers)
        labels_jigsaw.append(perms+1)
        #recompose the new image
        img_recomposed = Image.new('RGB',img.size);
        pi = 0
        for i in range(0,width,piece_size):
            for j in range(0,height,piece_size):
                index1 = numbers[pi]
                img_recomposed.paste(parts[index1],(i,j))
                pi=pi+1
        #so now we have the image recomposed the vector of permutations and we calculate the label
        imgs_jigsaw.append(img_recomposed)
        
    return name_jigsaw,imgs_jigsaw,permutations_jigsaw,labels_jigsaw


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None,betaJigen=0.2):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.betaJigen = betaJigen
        #let's take beta part of training images and use them for jigsaw puzzle
        self.jigsaw_names,self.jigsaw_imgs,self.jigsaw_permutations,self.jigsaw_labels = prepare_jigsaw_data(names,betaJigen,self.data_path);
        
        

    def __getitem__(self, index):
        
        if(index < len(self.names)): 
            framename = self.data_path + '/' + self.names[index]
            img = Image.open(framename).convert('RGB')
            img = self._image_transformer(img)
            #case of not scrumbled image
            return img, int(self.labels[index]), int (0)
        else:
            index = index - len(self.names)
            #case of scrumbled image
            return self.jigsaw_imgs[index], int(self.labels[index]),int(self.jigsaw_labels[index])
        

    def __len__(self):
        return len(self.names)
    
    



class TestDataset(Dataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):

        if(index < len(self.names)): 
            framename = self.data_path + '/' + self.names[index]
            img = Image.open(framename).convert('RGB')
            img = self._image_transformer(img)
            #case of not scrumbled image
            return img, int(self.labels[index]), int (0)
        else:
            index = index - len(self.names)
            #case of scrumbled image
            return self.jigsaw_imgs[index], int(self.labels[index]),int(self.jigsaw_labels[index])


