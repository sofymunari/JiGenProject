import numpy as np
import torch
import math
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from itertools import product,permutations
from random import sample, random,randint
from scipy.spatial import distance


#jigsaw_permutations has the top scrumbles parts-> labels = position +1

def calculate_possible_permutations():
    vec = [0,1,2,3,4,5,6,7,8]
    top_30 = []
    top_30_humming = np.zeros(30)
    res = list(permutations(vec))
    for v in res:
        #calculate hamming distance
        dist = distance.hamming(vec, v)
        for i in range(0,30):
            if dist > top_30_humming[i]:
                top_30_humming[i]=dist
                top_30.insert(i,v)
    #now i have top 30 permutations saved 
    global jigsaw_permutations
    jigsaw_permutations = top_30
        
        


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

def prepare_oddOneOut_data(names,percent,path):
    samples = len(names)
    amount = int(samples*percent)
    random_index = sample(range(samples),amount)
    name_jigsaw = [names[k] for k in random_index]
    imgs_jigsaw = []
    labels_jigsaw = []
    for index in random_index:
        framename = path +'/'+names[index]
        img = Image.open(framename).convert('RGB')
        width, height = img.size
        #i want 3x3 grid

        piece_size_w = int(math.ceil(width / 3))
        piece_size_h = int(math.ceil(height / 3))
        width_steps = range(0, width, piece_size_w)
        height_steps = range(0, height, piece_size_h)
        boxes = ((i, j, i+piece_size_w, j+piece_size_h)
                 for i, j in product(width_steps, height_steps))
        parts = [img.crop(box) for box in boxes]
        oddOne = randint(1,9)
        #0-> non odd one inserted
        #1-9 position of the odd one inserted
        labels_jigsaw.append(oddOne)
        #get a piece randomly from another image
        random_chosen_img_odd = randint(0,samples-1)
        framename_chosen = path +'/'+names[random_chosen_img_odd]
        img_odd = Image.open(framename_chosen).convert('RGB')
        w_odd, h_odd = img_odd.size
        #i want 3x3 grid

        piece_size_w_odd = int(math.ceil(w_odd / 3))
        piece_size_h_odd = int(math.ceil(h_odd / 3))
        width_steps_odd = range(0, w_odd, piece_size_w_odd)
        height_steps_odd = range(0, h_odd, piece_size_h_odd)
        boxes_odd = ((i_odd, j_odd, i_odd+piece_size_w_odd, j_odd+piece_size_h_odd)
                 for i_odd, j_odd in product(width_steps_odd, height_steps_odd))
        random_piece_of_img = randint(0,8)
        parts_to_insert = [img_odd.crop(box_odd) for box_odd in boxes_odd]
        the_part_to_insert = parts_to_insert[random_piece_of_img]
        pi = 0
        img_recomposed = Image.new('RGB',img.size)
        for i in range(0,width,piece_size_w):
            for j in range(0,height,piece_size_h):
                if(pi!=oddOne -1):
                    img_recomposed.paste(parts[pi],(i,j))
                else:
                    img_recomposed.paste(the_part_to_insert,(i,j))
                pi=pi+1
        imgs_jigsaw.append(img_recomposed)
    return name_jigsaw,imgs_jigsaw,labels_jigsaw
        
        
    


def prepare_rotation_data(names,percent,path):
    samples = len(names)
    amount = int(samples*percent)
    random_index = sample(range(samples),amount)
    name_jigsaw = [names[k] for k in random_index]
    imgs_jigsaw = []
    labels_jigsaw = []
    for index in random_index:
        framename = path +'/'+names[index]
        img = Image.open(framename).convert('RGB')
        #get random rotation to apply
        #0-> no rotation
        #1-> 90 degrees rotation
        #2-> 180 degrees rotation
        #3-> 270 degrees rotation
        
        rotation = randint(1,3)
        if rotation == 1:
            theRotation = Image.ROTATE_90
        elif rotation == 2:
            theRotation = Image.ROTATE_180 
        elif rotation == 3:
            theRotation = Image.ROTATE_270
        labels_jigsaw.append(rotation)
        rotatedImage  = img.transpose(theRotation)
        imgs_jigsaw.append(rotatedImage)
    return name_jigsaw,imgs_jigsaw,labels_jigsaw
        
    

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
        
        width, height = img.size
        #i want 3x3 grid

        piece_size_w = int(math.ceil(width / 3))
        piece_size_h = int(math.ceil(height / 3))
        width_steps = range(0, width, piece_size_w)
        height_steps = range(0, height, piece_size_h)
        boxes = ((i, j, i+piece_size_w, j+piece_size_h)
                 for i, j in product(width_steps, height_steps))
        parts = [img.crop(box) for box in boxes]
        #lets get one of the permutations
        perms = randint(0, 29)
        numbers = jigsaw_permutations[perms]
        permutations_jigsaw.append(numbers)
        labels_jigsaw.append(perms+1)
        #recompose the new image
        img_recomposed = Image.new('RGB',img.size)
        pi = 0
        for i in range(0,width,piece_size_w):
            for j in range(0,height,piece_size_h):
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
    def __init__(self, names, labels, path_dataset,img_transformer=None,betaJigen=0.2,rotation=False,oddOneOut= False):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.betaJigen = betaJigen
        #let's take beta part of training images and use them for jigsaw puzzle
        if rotation == True:
            #no jigsaw but we keep those names to avoid using more variables
            self.jigsaw_names,self.jigsaw_imgs,self.jigsaw_labels = prepare_rotation_data(names,betaJigen,self.data_path)
        elif oddOneOut == True:
            self.jigsaw_names,self.jigsaw_imgs,self.jigsaw_labels = prepare_oddOneOut_data(names,betaJigen,self.data_path);
        else:
            self.jigsaw_names,self.jigsaw_imgs,self.jigsaw_permutations,self.jigsaw_labels = prepare_jigsaw_data(names,betaJigen,self.data_path)
            
        

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
            img = self._image_transformer(self.jigsaw_imgs[index])
            return img, int(self.labels[index]),int(self.jigsaw_labels[index])
        

    def __len__(self):
        return len(self.names)+len(self.jigsaw_names)
    

    
    



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
            img = self._image_transformer(self.jigsaw_imgs[index])
            return img, int(self.labels[index]),int(self.jigsaw_labels[index])


