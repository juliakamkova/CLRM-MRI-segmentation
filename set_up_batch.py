#import numpy as np
#import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import utils 
import evaluation
import monai
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from pathlib import Path
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from box import Box

class Framework(pl.LightningModule):
    def __init__(self, net, params, n_cl):
        super().__init__()
        self.net = net
        self.params = params
        self.n_cl = n_cl
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=n_cl)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_cl)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []


    def forward(self, x):
        segmentation = self.net(x)
        return segmentation  

    def prepare_data(self):
        #set up correct data path
        train_images = sorted(Path(self.params.data.image).glob('train/*.nii*'))
        train_labels = sorted(Path(self.params.data.label).glob('train/*.nii*'))
        print('train images path',Path(self.params.data.image), train_images)
        #print('training dataset images' ,train_images)

        #print('training dataset images STR' , str(train_images))

        train_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(train_images, train_labels)
        ]
       
       # set deterministic training for reproducibility
        set_determinism(seed=0)


        print(len(train_dicts))
        
        transforms = []
        for class_params in self.params.train.augmentation:
            #print('checking which train augmentation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))
        train_transform = monai.transforms.Compose(transforms)
        
        
        valid_images = sorted(Path(self.params.data.image).glob('valid/*.nii*'))
        valid_labels = sorted(Path(self.params.data.label).glob('valid/*.nii*'))
        print('validation dataset images' ,valid_images)
        #print('validation dataset labels', valid_labels)

        valid_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(valid_images, valid_labels)
        ]
              
        transforms = []
        for class_params in self.params.valid.augmentation:
            #print('checking which train augmentation in validation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))   
        valid_transform = monai.transforms.Compose(transforms)
 
        # we use cached datasets - these are 10x faster than regular datasets
        self.training_ds = CacheDataset(data=train_dicts, transform=train_transform, cache_rate=1.0, num_workers=4)# num_workers=8)
        self.validation_ds = CacheDataset(data=valid_dicts, transform=valid_transform, cache_rate=1.0, num_workers=4)
        #self.test_ds = CacheDataset(data=valid_dicts, transform=valid_transform, num_workers=4)

        self.loss_function = utils.class_loader(self.params.loss)[0]
        print('from prepare data note: loss function that is used: ', self.loss_function)   
  
    def train_dataloader(self):
        self.training_dataloader = torch.utils.data.DataLoader(self.training_ds, batch_size=2, shuffle=True, num_workers=4, persistent_workers=True) #excessive worker creation might get DataLoader running slow or even freeze
        return self.training_dataloader

    def val_dataloader(self):
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_ds,batch_size=1 , num_workers=4, persistent_workers=True) #excessive worker creation might get DataLoader running slow or even freeze
        return self.validation_dataloader

    def configure_optimizers(self):
        name_optimizer = list(self.params.optimizer.keys())[0]
        self.lr = self.params.optimizer[name_optimizer].kwargs.lr
        optim = utils.class_loader(self.params.optimizer, extra_args=(self.net.parameters(),))[0]
        lr_dict = {
            "optimizer": optim,
            "monitor": 'val_loss',
           }
        # #print('from cofigure otimizer note: optimizer that is used: ', optim, 'check learning rate', self.lr, 'lr scedulare', lr_dict["scheduler"] )
        return lr_dict

    def training_step(self, batch, batch_idx):
        input = batch['image']
        mask = batch['label']
        output = self.forward(input)
        print('output size', output.size())

        loss = self.loss_function(output, mask)
        train_monai_dice = self.dice_metric(y_pred=output, y=mask)
        #self.log('train_batch_Dice', train_monai_dice)
        #self.log('train_batch_loss', train_loss)
        return {"loss": loss, "train_batch_dice": train_monai_dice}

    def validation_step(self, batch, batch_idx):
        input = batch['image']
        masks = batch['label']
        output = self.forward(input)

        val_loss = self.loss_function(output, masks)
        print('val loss inside the lightning', val_loss)
        self.log('val_step_loss', val_loss)
        
        print('from validation step', masks.shape, output.shape)

        monai_dice = self.dice_metric ( y_pred=output, y=masks )
        print ( 'from validation step checking Dice', monai_dice, batch_idx )
        #self.log('val step Dice', monai_dice)
        d ={"val_batch_loss": val_loss, "val_batch_dice": monai_dice, "val_number" : len(output)}
        self.validation_step_outputs.append(d)
        return 


    def on_validation_epoch_end(self):
        val_dice, val_loss, num_items = 0, 0, 0
        for output in self.validation_step_outputs:
            print('checking the output dice', output["val_batch_dice"].size())
            val_dice += output["val_batch_dice"].sum().item()
            val_loss += output["val_batch_loss"].sum().item()
            num_items += output["val_number"]

        #mean_val_dice = self.dice_metric.aggregate().item() #i think this belongs to patch version of the code
        #self.dice_metric.reset()

        mean_val_dice = torch.tensor(val_dice / num_items)
        mean_val_loss = torch.tensor(val_loss / num_items)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        #self.log('val_epoch_Dice', mean_val_dice)
        self.log('val_epoch_Loss', mean_val_loss)
        self.log('Best validation Dice', self.best_val_dice)

        print(
            f"current epoch: {self.current_epoch} current val mean dice: {mean_val_dice:.4f} current val loss {mean_val_loss:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} at epoch: {self.best_val_epoch:.4f}"
        )

    


    # def test_step(self, batch, batch_idx):
    #     print('we are in the beggining of the test step')
    #     input = batch['image']
    #     mask = batch['label']
    #     output = self.forward(input)
    #     loss = self.loss_function(output, mask)
    #     self.log('test_loss', loss)
    #     output1 = F.one_hot(output, num_classes = 3)
    #     monai_dice = self.dice_metric(y_pred=output1,  #!TODO
    #                     y=mask)
    #
    #     prediction = output[0,...]
    #     prediction = prediction.detach().cpu().argmax(dim=0).float()
    #     #evaluation.save_batch(prediction, 'test_try.nii')
    #     print('test_loss', loss, 'test_dice', monai_dice)
    #     return {"test_loss": loss, "test_dice": monai_dice}


# def test_end(self, outputs):
#        avg_loss= torch.stack([x['test_loss'] for x in outputs]).mean()
#        tensorboard_logs = {'xxavg_loss_for_test': avg_loss}
#        return {'xtest_loss': avg_loss, 'log': tensorboard_logs}
