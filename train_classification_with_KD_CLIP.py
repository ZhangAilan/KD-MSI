import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import glob
from torch.utils.data._utils.collate import default_collate

from core.networks import Classifier_Siamese
from core.WS_dataset import WSCDDataSet, WSCDDataSet_iou_evaluate,Iterator
from tools.general.io_utils import create_directory, str2bool
from tools.general.time_utils import Timer
from tools.ai.log_utils import log_print, Average_Meter
from tools.ai.optim_utils import PolyOptimizer
from tools.ai.torch_utils import load_model, set_seed, make_cam, save_model, get_numpy_from_tensor,get_learning_rate_from_optimizer,accuracy
from tools.ai.evaluate_utils import Calculator_For_mIoU, Calculator_For_F1
from CLIP.clip import create_model
from core.extract_feature import encode_text_for_change_detection, get_feature_dinov3
from core.adapter import DinoToClipProjector
from core.loss import FocalLoss

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix
parser.add_argument('--teacher', default='abs', type=str)
parser.add_argument('--student', default='minus', type=str)
###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_epoch', default=20, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=256, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', required=True, type=str)

def accuracy(pred, y):
    correct = sum(row.all().int().item() for row in (pred.ge(0) == y))
    n = y.shape[0]
    return correct / n

def custom_collate(batch):
    # Separate the batch into components
    pre_images, post_images, labels, sentences_list = zip(*batch)

    # Collate the images and labels normally
    pre_images = default_collate(pre_images)
    post_images = default_collate(post_images)
    labels = default_collate(labels)

    # Keep sentences as a list of lists (each element is a list of sentences for an image)
    # This prevents the default collate from transposing the sentences

    return pre_images, post_images, labels, sentences_list


def load_clip_model(device,ckpt_path):
    """Load the CLIP model."""
    # -------- 2. CLIP 加载 --------
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=512,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        ckpt_path=ckpt_path,
    )
    clip_model.to(device).eval()
    print("CLIP model loaded.")
    return clip_model


def load_dinov3_model(device,ckpt_path):
    """Load the DINOv3 model."""
    # -------- 3. DINOv3 加载 --------
    repo_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dinov3"
    )
    dinov3_model = torch.hub.load(
        repo_dir,
        "dinov3_vitl16",
        source="local",
        weights=ckpt_path,
    )
    dinov3_model.to(device).eval()
    print("DINOv3 model loaded.")
    return dinov3_model


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')

    log_path = log_dir + f'{args.tag}.txt'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.tag))
    log_func()

    jsonfile=glob.glob(args.data_dir + '/*.json')
    jsonfile=jsonfile[0]
    train_dataset = WSCDDataSet(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                 list_file=args.data_dir+'/list/train_label.txt',
                                 img_size=args.image_size,jsonpath=jsonfile)

    valid_dataset = WSCDDataSet_iou_evaluate(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                             mask_folder=args.data_dir+'/label',
                                 list_file=args.data_dir+'/list/val_label.txt',
                                 img_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True,collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    # 强制使用单显卡，如果有 GPU 则用 cuda:0，否则用 cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier_Siamese(args.architecture, 1, args.mode, args.teacher)
    model2 = Classifier_Siamese(args.architecture, 1, args.mode, args.student)
    
    param_groups = model.get_parameter_groups(print_fn=None)
    param_groups2 = model2.get_parameter_groups(print_fn=None)
    
    # 统一移动到设备并设为训练模式
    model.to(device).train()
    model2.to(device).train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    load_model_fn = lambda: load_model(model2, model_path, parallel=False)
    save_model_fn = lambda: save_model(model2, model_path, parallel=False)

    # Load models
    dino_model = load_dinov3_model(device,'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    clip_model = load_clip_model(device,'ViT-L-14-336px.pt')

    #DINO CLIP Adaptor
    projector_dino_clip=nn.Linear(1024, 768, bias=False).cuda()
    # projector_dino_clip = DinoToClipProjector(in_dim=1024, out_dim=768).to(device)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.BCEWithLogitsLoss().cuda()
    loss_focal=FocalLoss()

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},

        {'params': param_groups2[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups2[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups2[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups2[3], 'lr': 20*args.lr, 'weight_decay': 0},

        {'params': projector_dino_clip.parameters(), 'lr': 10 * args.lr, 'weight_decay': args.wd},

    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'class_loss', 'loss_cross', 'accuracy'])

    best_acc1 = -1
    best_f1 = -1
    thresholds = list(np.arange(0.10, 1, 0.05))
    
    def evaluate(loader):
        model2.eval()
        eval_timer.tik()

        valid_meter = Average_Meter(['val_loss','val_acc'])
        meter_dic = {th : Calculator_For_mIoU() for th in thresholds}
        f1_meter_dic = {th : Calculator_For_F1() for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (imageA, imageB, labels,gt_masks) in enumerate(loader):
                imageA = imageA.cuda()
                imageB = imageB.cuda()
                labels =  labels.cuda()                                                  
                logits, features = model2(imageA,imageB)              
                loss = class_loss_fn(logits, labels).mean()
                acc = accuracy(logits, labels)
                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = torch.sigmoid(features)*mask
                valid_meter.add({'val_loss': loss.item(),'val_acc':acc})

                for batch_index in range(imageA.size()[0]):
                        # c, h, w -> h, w, c
                        cam = get_numpy_from_tensor(cams[batch_index]).transpose((1, 2, 0))
                        cam = cv2.resize(cam,(256,256),interpolation=cv2.INTER_NEAREST)
                        cam = cam.reshape(cam.shape[0],cam.shape[1],1)
                        gt_mask = get_numpy_from_tensor(gt_masks[batch_index])                      
                        h, w,c = cam.shape
                        gt_mask = cv2.resize(gt_mask, (h,w), interpolation=cv2.INTER_NEAREST)
                        for th in thresholds:
                            bg = np.ones_like(cam[:, :, 0]) * th
                            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)
                            meter_dic[th].add(pred_mask, gt_mask)
                            f1_meter_dic[th].add(pred_mask, gt_mask)
                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        model2.train()

        best_th = 0.0
        best_mIoU = 0.0
        best_f1 = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            f1 = f1_meter_dic[th].get(clear=True)
            if best_mIoU < mIoU_foreground:
                best_th = th
                best_mIoU = mIoU_foreground
                best_f1 = f1

        return valid_meter.get(clear=True), best_th, best_mIoU, best_f1

    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)                                                          

    no_change_text_feature = encode_text_for_change_detection(clip_model=clip_model, device=device)
    for iteration in range(max_iteration):
        imageA, imageB, labels,sentences = train_iterator.get()
        imageA, imageB, labels = imageA.cuda(), imageB.cuda(), labels.cuda()
        # print(imageA.shape)    #[8, 3, 256, 256]
        # print(labels.shape)    #[8, 1]
        # print(len(sentences))  #[8]

        #文字特征
        text_features = torch.zeros(len(sentences), 768, 2).to(device)
        with torch.no_grad():
            for i in range(len(sentences)):
                #text branch
                if labels[i]==0: #此时未发生变化
                    text_features[i] = no_change_text_feature
                else: #此时发生变化
                    text_features[i] = encode_text_for_change_detection(clip_model, device, batch_change_sentences=sentences[i])     
        # print(text_features.shape) #[8, 768, 2]

        #视觉特征
        _,patch_tokens_A=get_feature_dinov3(imageA, device, dino_model)
        _,patch_tokens_B=get_feature_dinov3(imageB, device, dino_model)
        # print(patch_tokens_A[0].shape) #[8, 256, 1024]

        #文字与视觉特征适配
        cross_modal_features_list=[]
        for i in range(4):
            pathch_features_A=projector_dino_clip(patch_tokens_A[i])
            pathch_features_B=projector_dino_clip(patch_tokens_B[i])
            patch_features=torch.abs(torch.sub(pathch_features_A, pathch_features_B))
            #归一化
            eps=1e-6
            patch_features=patch_features/(patch_features.norm(dim=-1, keepdim=True)+eps)
            # print(patch_features.shape) #[8, 256, 768]
            #跨模态
            LOGITS_SCALE=100.0
            cross_modal_features=LOGITS_SCALE*patch_features@text_features
            # print(cross_modal_features.shape) #[8, 256, 2]
            B,N,C=cross_modal_features.shape
            H,W=int(np.sqrt(N)),int(np.sqrt(N))
            cross_modal_features = cross_modal_features.view(B, H, W, C)
            cross_modal_features = cross_modal_features.permute(0, 3, 1, 2)
            # print(cross_modal_features.shape) #[8, 2, 16, 16]
            cross_modal_features = torch.softmax(cross_modal_features, dim=1)
            cross_modal_features_list.append(cross_modal_features)
        cross_modal_features = torch.mean(torch.stack(cross_modal_features_list, dim=0), dim=0) # [8, 2, 16, 16]

        #生成CAM
        logits , features1= model(imageA,imageB)
        logits2 ,features2 = model2(imageA,imageB)

        cam = make_cam(features1)*labels.unsqueeze(2).unsqueeze(3)
        cam1 = cam.clone().detach()  #教师cam
        cam2 = F.sigmoid(features2)  #学生cam 
        # print(cam2.shape) #[8, 1, 16, 16]

        loss_kd = nn.MSELoss()(cam2,cam1)
        class_loss = class_loss_fn(logits, labels).mean()

        #计算跨模态损失
        # loss_cross=nn.MSELoss()(cam2,cross_modal_features[:, 1:2, :, :])
        loss_cross=loss_focal(cross_modal_features,cam2)

        acc1= accuracy(logits, labels)

        #加上跨模态损失
        loss = class_loss + 10*loss_kd+10*loss_cross
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'class_loss' : class_loss.item(),
            'loss_cross' : loss_cross.item(),
            'accuracy':acc1
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, class_loss, loss_cross, acc1 = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'class_loss' : class_loss,
                'loss_cross' : loss_cross,
                'acc1':acc1,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                loss_cross={loss_cross:.4f}, \
                acc1={acc1:.4f},\
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/class_loss', class_loss, iteration)
            writer.add_scalar('Train/loss_cross', loss_cross, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
            writer.add_scalar('Train/acc1', acc1, iteration)
        
        ################################################################################################
        # Evaluation
        ################################################################################################
        # if True:
        if (iteration + 1) % val_iteration == 0:
            (valid_loss,val_acc1),threshold, mIoU, f1 = evaluate(valid_loader)
            
            if best_acc1 == -1 or best_acc1 < val_acc1:
                best_acc1 = val_acc1

            if best_f1 == -1 or best_f1 < f1:
                best_f1 = f1

                save_model_fn()
                log_func('[i] save model (best F1)')

            data = {
                'iteration' : iteration + 1,
                'valid_loss' : valid_loss,
                'best_acc1' : best_acc1,
                'val_acc1' : val_acc1,
                'train_mIoU':mIoU,
                'threshold':threshold,
                'f1':f1,
                'best_f1':best_f1,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)

            
            log_func('[i] \
                iteration={iteration:,}, \
                valid_loss={valid_loss:.4f}, \
                val_acc1={val_acc1:.4f}%,\
                best_acc1={best_acc1:.4f}%, \
                threshold={threshold:.2f}, \
                train_mIoU={train_mIoU:.2f}%, \
                f1={f1:.2f}%, \
                best_f1={best_f1:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            
            writer.add_scalar('Evaluation/valid_loss', valid_loss, iteration)
            writer.add_scalar('Evaluation/valid_acc1', val_acc1, iteration)
            writer.add_scalar('Evaluation/f1', f1, iteration)
            writer.add_scalar('Evaluation/best_f1', best_f1, iteration)
            writer.add_scalar('Evaluation/best_acc1', best_acc1, iteration)
    

    writer.close()

    print(args.tag)