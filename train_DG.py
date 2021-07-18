import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
from data.DatasetLoader import calculate_possible_permutations
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np

Alpha = 0.1 #tot a caso, percentuale dell'errore del jigsaw
Beta = 0.5 #tot a caso, percentuale del batch usato per input image del jigsaw
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--path_dataset", default="/home/silvia/Jigen_AIMLProject/", help="Path where the dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="alexnet", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    
    #batch of data used for jigsaw puzzle
    parser.add_argument("--betaJigen", type=float, default=0.2, help="percentage of data used for jigsaw puzzle")
    parser.add_argument("--rotation", type=bool, default= False, help="are you running Rotation classfication self supervised task?" )
    parser.add_argument("--oddOneOut", type=bool, default= False, help="are you running Odd One Out classfication self supervised task?" )
    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.alpha_jigsaw_weight = 0.5
        self.alpha_odd_weight = 0.5
        self.alpha_rotation_weight = 0.5
        self.args = args
        self.device = device
        self.betaJigen = args.betaJigen
        
        model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=31,odd_classes=10,rotation_classes = 4)
        #if args.rotation== True:
        #    model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=4)
        #elif args.oddOneOut == True:
        #    model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=10)
        #else:
        #    model = model_factory.get_network(args.network)(classes=args.n_classes,jigsaw_classes=31)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_val_dataloader(args)


        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

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
        for it, (data, class_l, jigsaw_l,self_sup_task) in enumerate(self.source_loader):
            #source_loader is only data for training
            data, class_l,jigsaw_l,self_sup_task = data.to(self.device), class_l.to(self.device),jigsaw_l.to(self.device),self_sup_task.to(self.device)

            self.optimizer.zero_grad()
            
            class_logit,jigsaw_logit,odd_logit,rotation_logit = self.model(data) #label from model
            #evaluate jigsaw mistake
            jigsaw_loss = criterion(jigsaw_logit[(self_sup_task==0)|(self_sup_task == 3)],jigsaw_l[(self_sup_task==0)|(self_sup_task == 3)])
            
            if self.args.oddOneOut == True:
            
                odd_loss = criterion(odd_logit[(self_sup_task==1)|(self_sup_task == 3)],jigsaw_l[(self_sup_task==1)|(self_sup_task == 3)])
            else:
                odd_loss = 0
            
            if self.args.rotation == True:
                rotation_loss = criterion(rotation_logit[(self_sup_task==2)|(self_sup_task == 3)],jigsaw_l[(self_sup_task==2)|(self_sup_task == 3)])
            else:
                rotation_loss = 0
            
         
            
            #for classification we evaluate the loss only for the not scrumbled images
            class_loss = criterion(class_logit[jigsaw_l == 0], class_l[jigsaw_l == 0])
            
            _, jigsaw_pred = jigsaw_logit[(self_sup_task==0)|(self_sup_task == 3)].max(dim=1)
            
            if self.args.oddOneOut ==  True:
                _, odd_pred = odd_logit[(self_sup_task==1)|(self_sup_task == 3)].max(dim=1)
            
            if self.args.rotation == True:
                _, rotation_pred = rotation_logit[(self_sup_task==2)|(self_sup_task == 3)].max(dim=1)
                
            _, cls_pred = class_logit.max(dim=1)

            loss = class_loss + self.alpha_jigsaw_weight * jigsaw_loss+ self.alpha_odd_weight*odd_loss +self.alpha_rotation_weight*rotation_loss

            loss.backward()

            self.optimizer.step()

            if self.args.oddOneOut == True and self.args.rotation == True:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Odd Loss": odd_loss.item(),"Rotation Loss": rotation_loss.item() },
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0)|(self_sup_task == 3)].data).item(),"Odd Accuracy ": torch.sum(odd_pred == jigsaw_l[(self_sup_task==1)|(self_sup_task == 3)].data).item(),"Rotation Accuracy ": torch.sum(rotation_pred == jigsaw_l[(self_sup_task==2)|(self_sup_task == 3)].data).item()},
                                data.shape[0])
            elif self.args.oddOneOut == True and self.args.rotation == False:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Odd Loss": odd_loss.item() },
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0)|(self_sup_task == 3)].data).item(),"Odd Accuracy ": torch.sum(odd_pred == jigsaw_l[(self_sup_task==1)|(self_sup_task == 3)].data).item()},
                                data.shape[0])
            elif self.args.oddOneOut == False and self.args.rotation == True:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item(),"Rotation Loss": rotation_loss.item() },
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0)|(self_sup_task == 3)].data).item(),"Rotation Accuracy ": torch.sum(rotation_pred == jigsaw_l[(self_sup_task==2)|(self_sup_task == 3)].data).item()},
                                data.shape[0])
            else:
                self.logger.log(it, len(self.source_loader),
                                {"Class Loss ": class_loss.item(), "Jigsaw Loss": jigsaw_loss.item()},
                                {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item(),"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_l[(self_sup_task==0)|(self_sup_task == 3)].data).item()},
                                data.shape[0])
                
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit,odd_loss,rotation_loss,odd_logit,rotation_logit


        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct,jigsaw_correct,odd_correct,rotation_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                jigsaw_acc = float(jigsaw_correct) / total
                odd_acc = float(odd_correct)/total
                rotation_acc = float(rotation_correct)/total
                acc = (class_acc + jigsaw_acc + odd_acc + rotation_acc) /self.nTasks
                self.logger.log_test(phase, {"Classification Accuracy": acc})
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