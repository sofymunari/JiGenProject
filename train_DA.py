import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
import torch.nn.functional as func
from models import model_factory
from data.DatasetLoader import calculate_possible_permutations
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
import itertools


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--path_dataset", default="/home/silvia/Jigen_AIMLProject/", help="Path where the dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="resnet18", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--betaJigen", type=float, default=0.2, help="percentage of data used for jigsaw puzzle")
    parser.add_argument("--rotation", type=bool, default=False, help="implementing rotation as self supervised task")
    parser.add_argument("--oddOneOut", type=bool, default=False, help="implementing Odd One Out as self supervised task")
    return parser.parse_args()

def entropy_loss(x):
    return torch.sum(-func.softmax(x,1) * func.log_softmax(x,1),1).mean()

class Trainer:
    def __init__(self, args, device):
        self.alpha_jigsaw_weight = 0.5
        self.alpha_jigsaw_weight_target = 0.5
        self.alpha_rotation_weight = 0.5
        self.alpha_rotation_weight_target = 0.5
        self.alpha_odd_weight = 0.5
        self.alpha_odd_weight_target = 0.5
        
        self.entropi_ni = 0.1
        self.args = args
        self.device = device
        self.betaJigen = args.betaJigen
        #if args.rotation == True:
        model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=31,odd_classes=10,rotation_classes = 4)
       # elif args.oddOneOut == True:
       #     model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=10)
        #else:
        #    model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=31)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_val_dataloader(args)
        self.targetAsSource_loader = data_helper.get_trainTargetAsSource_dataloader(args);


        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset)+len(self.targetAsSource_loader), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)
       
        self.n_classes = args.n_classes
        if args.oddOneOut== True and args.rotation == True:
            self.nTasks = 4
        elif args.oddOneOut== True or args.rotation == True:
            self.nTasks = 3
        else:
            self.nTasks = 2

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, class_l, jigsaw_l,self_sup_task), (data_target, class_l_target, jigsaw_l_target,self_sup_task_target)) in enumerate(zip(self.source_loader, itertools.cycle(self.targetAsSource_loader))):
        #for it,  in enumerate(self.source_loader):

            data, class_l,jigsaw_l,self_sup_task,data_target,jigsaw_l_target,self_sup_task_target = data.to(self.device), class_l.to(self.device),jigsaw_l.to(self.device),self_sup_task.to(self.device),data_target.to(self.device),jigsaw_l_target.to(self.device),self_sup_task_target.to(self.device)

            self.optimizer.zero_grad()

            class_logit,jigsaw_logit,odd_logit,rotation_logit = self.model(data) #label from model
            class_logit_target,jigsaw_logit_target,odd_logit_target,rotation_logit_target = self.model(data_target)
            jigsaw_loss = criterion(jigsaw_logit[(self_sup_task == 0) | (self_sup_task == 3)],jigsaw_l[(self_sup_task == 0) | (self_sup_task == 3)])
            jigsaw_loss_target = criterion(jigsaw_logit_target[(self_sup_task_target == 0) | (self_sup_task_target == 3)],jigsaw_l_target[(self_sup_task_target == 0) | (self_sup_task_target == 3)])
            
            if self.args.oddOneOut == True :
                odd_loss = criterion(odd_logit[(self_sup_task == 1) | (self_sup_task == 3)],jigsaw_l[(self_sup_task == 1) | (self_sup_task == 3)])
                odd_loss_target = criterion(odd_logit_target[(self_sup_task_target == 1) | (self_sup_task_target == 3)],jigsaw_l_target[(self_sup_task_target == 1) | (self_sup_task_target == 3)])
            else:
                odd_loss = 0
                odd_loss_target = 0
                
            if self.args.rotation == True :
                rotation_loss = criterion(rotation_logit[(self_sup_task == 2) | (self_sup_task == 3)],jigsaw_l[(self_sup_task == 2) | (self_sup_task == 3)])
                rotation_loss_target = criterion(rotation_logit_target[(self_sup_task_target == 2 )| (self_sup_task_target == 3)],jigsaw_l_target[(self_sup_task_target == 2) | (self_sup_task_target == 3)])
            else:
                rotation_loss = 0
                rotation_loss_target = 0
            
            entropy_loss_target = entropy_loss(class_logit_target[jigsaw_l_target == 0])
            #traing classifier only on images not scrumbled and not in target!
            class_loss = criterion(class_logit[jigsaw_l == 0], class_l[jigsaw_l == 0])
            
            
            _, jigsaw_pred = jigsaw_logit[(self_sup_task == 0) | (self_sup_task == 3)].max(dim=1)
            _, jigsaw_pred_target = jigsaw_logit_target[(self_sup_task_target == 0) | (self_sup_task_target == 3)].max(dim=1)
            
            if self.args.oddOneOut == True:
                _, odd_pred = odd_logit[(self_sup_task == 1) | (self_sup_task == 3)].max(dim=1)
                _, odd_pred_target = odd_logit_target[(self_sup_task_target == 1) | (self_sup_task_target == 3)].max(dim=1)
            
            if self.args.rotation == True:
                _, rotation_pred = rotation_logit[(self_sup_task == 2 )| (self_sup_task == 3)].max(dim=1)
                _, rotation_pred_target = rotation_logit_target[(self_sup_task_target == 2) | (self_sup_task_target == 3)].max(dim=1)
            
            _, cls_pred = class_logit.max(dim=1)
            
            loss = class_loss + self.alpha_jigsaw_weight * jigsaw_loss+ self.alpha_jigsaw_weight_target * jigsaw_loss_target + self.entropi_ni * entropy_loss_target + odd_loss * self.alpha_odd_weight + odd_loss_target * self.alpha_odd_weight_target + rotation_loss * self.alpha_rotation_weight + rotation_loss_target * self.alpha_rotation_weight_target

            loss.backward()

            self.optimizer.step()

            if self.args.oddOneOut == True and self.args.rotation == True:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Jigsaw Target Loss": jigsaw_loss_target.item(), "Odd Loss": odd_loss.item(), "Odd Loss Target": odd_loss_target.item(), "Rotation Loss":rotation_loss.item(), "Rotation Loss Target":rotation_loss_target.item()},
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0) | (self_sup_task == 3)].data).item(), "Jigsaw Accuracy Target": torch.sum(jigsaw_pred_target == jigsaw_l_target[(self_sup_task_target==0) | (self_sup_task_target == 3)].data).item(),"Odd Accuracy ": torch.sum(odd_pred == jigsaw_l[(self_sup_task==1) | (self_sup_task == 3)].data).item(), "Odd Accuracy Target": torch.sum(odd_pred_target == jigsaw_l_target[(self_sup_task_target==1) | (self_sup_task_target == 3)].data).item(),"Rotation Accuracy ": torch.sum(rotation_pred == jigsaw_l[(self_sup_task==2) | (self_sup_task == 3)].data).item(), "Rotation Accuracy Target": torch.sum(rotation_pred_target == jigsaw_l_target[(self_sup_task_target==2) | (self_sup_task_target == 3)].data).item()},
                                data.shape[0])
                
            elif self.args.oddOneOut == True and self.args.rotation == False:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Jigsaw Target Loss": jigsaw_loss_target.item(), "Odd Loss": odd_loss.item(), "Odd Loss Target": odd_loss_target.item()},
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0) | (self_sup_task == 3)].data).item(), "Jigsaw Accuracy Target": torch.sum(jigsaw_pred_target == jigsaw_l_target[(self_sup_task_target==0) | (self_sup_task_target == 3)].data).item(),"Odd Accuracy ": torch.sum(odd_pred == jigsaw_l[(self_sup_task==1) | (self_sup_task == 3)].data).item(), "Odd Accuracy Target": torch.sum(odd_pred_target == jigsaw_l_target[(self_sup_task_target==1) | (self_sup_task_target == 3)].data).item()},
                                data.shape[0])
                
            elif self.args.oddOneOut == False and self.args.rotation == True:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Jigsaw Target Loss": jigsaw_loss_target.item(),"Rotation Loss":rotation_loss.item(), "Rotation Loss Target":rotation_loss_target.item()},
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0) | (self_sup_task == 3)].data).item(), "Jigsaw Accuracy Target": torch.sum(jigsaw_pred_target == jigsaw_l_target[(self_sup_task_target==0) | (self_sup_task_target == 3)].data).item(),"Rotation Accuracy ": torch.sum(rotation_pred == jigsaw_l[(self_sup_task==2) | (self_sup_task == 3)].data).item(), "Rotation Accuracy Target": torch.sum(rotation_pred_target == jigsaw_l_target[(self_sup_task_target==2) | (self_sup_task_target == 3)].data).item()},
                                data.shape[0])
            else:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Jigsaw Target Loss": jigsaw_loss_target.item()},
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0) | (self_sup_task == 3)].data).item(), "Jigsaw Accuracy Target": torch.sum(jigsaw_pred_target == jigsaw_l_target[(self_sup_task_target==0) | (self_sup_task_target == 3)].data).item()},
                                data.shape[0])
                
            del loss, class_loss, jigsaw_loss,rotation_loss,rotation_loss_target,odd_loss,odd_loss_target,jigsaw_loss_target, jigsaw_logit, class_logit,class_logit_target,jigsaw_logit_target,odd_logit,odd_logit_target,rotation_logit,rotation_logit_target
        
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct,jigsaw_correct,odd_correct,rotation_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                jigsaw_acc = float(jigsaw_correct) / total
                odd_acc = float(odd_correct)/total
                rotation_acc = float(rotation_correct)/total
                
                acc = (class_acc + jigsaw_acc + odd_acc +rotation_acc)/self.nTasks
                self.logger.log_test(phase, {"Class Accuracy": acc})
                self.results[phase][self.current_epoch] = acc


    def do_test(self, loader):
        class_correct = 0
        jigsaw_correct = 0
        odd_correct = 0
        rotation_correct = 0
        for it, (data, class_l, jigsaw_l,self_sup_task) in enumerate(loader):
            data, class_l,jigsaw_l,self_sup_task = data.to(self.device), class_l.to(self.device),jigsaw_l.to(self.device),self_sup_task.to(self.device)
            class_logit,jigsaw_logit,odd_logit,rotation_logit = self.model(data)
            
            
            _, jigsaw_pred = jigsaw_logit.max(dim=1)
            
            if self.args.oddOneOut == True:
                _, odd_pred = odd_logit.max(dim=1)
                odd_correct += torch.sum(odd_pred == jigsaw_l.data)
            if self.args.rotation == True:
                _, rotation_pred = rotation_logit.max(dim=1)
                rotation_correct += torch.sum(rotation_pred == jigsaw_l.data)
                
            _, cls_pred = class_logit.max(dim=1)
            
            jigsaw_correct += torch.sum(jigsaw_pred == jigsaw_l.data)
            
            
            
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct,jigsaw_correct,odd_correct,rotation_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
            self.scheduler.step()

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    calculate_possible_permutations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()