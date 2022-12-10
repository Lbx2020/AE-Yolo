
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ghost_yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    classes_path    = 'model_data/voc_classes.txt'
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6,7,8],[3, 4, 5], [0, 1, 2]]
    model_path      = 'model_data/yolov4_tiny_weights_voc.pth'
    input_shape     = [512, 512]
    mosaic              = True
    Cosine_lr           = False
    label_smoothing     = 0
    Init_Epoch          = 0
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-3
    Freeze_Train        = False
    num_workers         = 0
    train_annotation_path   = '2007_train.txt'
    val_split = 0.1
    with open(train_annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        a = {}
        for k, v in pretrained_dict.items():
            try:
                if np.shape(model_dict[k]) == np.shape(v):
                    a[k] = v
            except:
                pass

        model_dict.update(a)
        model.load_state_dict(model_dict)
        print('Finished!')

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/")


    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Init_Epoch
        end_epoch   = UnFreeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        train_dataset   = YoloDataset(lines[:num_train], input_shape, num_classes, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(lines[num_train:], input_shape, num_classes, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
