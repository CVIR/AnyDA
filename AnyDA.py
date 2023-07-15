import importlib
import os
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.setlogger import get_logger
import torch.nn as nn
from losses import info_max_loss
from utils.model_profiling import model_profiling
from utils.config import FLAGS
from utils.datasets import get_dataset
from itertools import cycle
from torch.autograd import Variable
from torch.autograd import Function

import warnings
warnings.filterwarnings("ignore") 

torch.multiprocessing.set_sharing_strategy('file_system')

global_step = 0
best_prec1 = 0
s_running_loss = 0.0
t_running_loss = 0.0
s_loss_vals = []
tpl_loss_vals = []
td_loss_vals = []
v_loss = []
acc_lst = []
# set log files
saved_path = os.path.join(FLAGS.log_dir, 'anyDA_{}-{}-{}_checkpoints'.format(FLAGS.dataset, FLAGS.model[7:], FLAGS.depth))
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}_{}to{}.log'.format('test' if FLAGS.test_only else 'train_lt',FLAGS.sdomain, FLAGS.tdomain)))

def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    return model

def get_optimizer(model):
    """get optimizer"""
    # all depthwise convolution (N, 1, x, x) has no weight decay
    # weight decay only on normal conv and fc
    if FLAGS.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                    momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                    weight_decay=FLAGS.weight_decay)
    return optimizer

        
def profiling_eda(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for res_mult in  sorted(FLAGS.resolution_list, reverse=True):
        model.apply(
                lambda r: setattr(r, 'res', res_mult))
        print("=========================Res:{}========================================".format(res_mult))
        for depth_mult in sorted(FLAGS.depth_mult_list, reverse=True):
            model.apply(
                lambda d: setattr(d, 'depth_mult', depth_mult))
            print("=========================Depth:{}========================================".format(depth_mult))
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))
                print("=========================Width:{}========================================".format(width_mult))
                print('Model profiling with width mult {}x:'.format(width_mult))
                verbose = width_mult == max(FLAGS.width_mult_list)
                model_profiling(
                    model, res_mult, res_mult,
                    verbose=getattr(FLAGS, 'model_profiling_verbose', verbose))

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        # Gradient Reversal Layer (GRL)
        return x

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

class AdversarialNetwork(nn.Module):
    """
    Domain Discriminator Network.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, x, lambd=1.0, use_grl=True):
        x_ = GradReverse(lambd)(x)
        y = self.main(x_)
        return y
        

def train(epoch, source_loader, target_loader, rd_loss, model_student, model_teacher, criterion, optimizer, lr_scheduler):

    t_start = time.time()
    model_student.train()

    combi_loader = zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader)
    tot_source_ce_loss = 0.0
    tot_rd_loss = 0.0
    tot_target_pl_loss = 0.0
    for batch_idx, data in enumerate(combi_loader):
        try:
            (source_data, target_data) = data
            source_input_list, source_target = source_data
            adv_label_source = torch.zeros_like(source_target).cuda().long()
            adv_label_target = torch.ones_like(source_target).cuda().long()
            source_target = source_target.cuda(non_blocking=True)
            if FLAGS.dataset != 'imagenet1k':    
                target_input_list, target_target = target_data
                model_teacher.apply(lambda d: setattr(d, 'depth_mult', sorted(FLAGS.depth_mult_range)[-1]))
                model_teacher.apply(lambda m: setattr(m, 'width_mult', sorted(FLAGS.width_mult_range)[-1])) 
                model_teacher.apply(lambda r: setattr(r, 'res', sorted(FLAGS.resolution_list)[-1]))            
                teacher_output_max = model_teacher(target_input_list[0].cuda(non_blocking=True),dom=1)
                teacher_output_lst = []
                inter_subnets = [[1,0.9,224],[1,1,192],[1,0.9,192],[0.5,1,224],[1,1,160],[0.5,0.9,224],[1,0.9,160],[0.5,1,192],[1,1,128],[0.5,0.9,192],[1,0.9,128],[0.5,1,160],[0.5,0.9,160],[0.5,1,128]]
                for snt in inter_subnets:
                    model_teacher.apply(lambda r: setattr(r, 'res', snt[2]))
                    model_teacher.apply(lambda m: setattr(m, 'width_mult', snt[1]))
                    model_teacher.apply(lambda d: setattr(d, 'depth_mult', snt[0]))
                    teacher_output_in = model_teacher(target_input_list[FLAGS.resolution_list.index(snt[2])].cuda(non_blocking=True),dom=1)
                    teacher_output_lst.append(teacher_output_in)              
    
            optimizer.zero_grad()
            #max-subnet
            subnets = [[sorted(FLAGS.depth_mult_range)[-1],sorted(FLAGS.width_mult_range)[-1],sorted(FLAGS.resolution_list)[-1]]]
            #2 random subnet   
            for _ in range((FLAGS.num_subnets-2)):
                subnets.append(random.choice([[1,0.9,224],[1,1,192],[1,0.9,192],[0.5,1,224],[1,1,160],[0.5,0.9,224],[1,0.9,160],[0.5,1,192],[1,1,128],[0.5,0.9,192],[1,0.9,128],[0.5,1,160],[0.5,0.9,160],[0.5,1,128]]))
            #min-subnet
            subnets.append([sorted(FLAGS.depth_mult_range)[0],sorted(FLAGS.width_mult_range)[0],sorted(FLAGS.resolution_list)[0]])
            subnet_out_lst = []
            info_loss = torch.tensor(0.0).cuda(non_blocking=True)
            for sn in subnets:
                
                model_student.apply(lambda d: setattr(d, 'depth_mult', sn[0]))
                model_student.apply(lambda m: setattr(m, 'width_mult', sn[1])) 
                model_student.apply(lambda r: setattr(r, 'res', sn[2]))

                if sn[0]==sorted(FLAGS.depth_mult_range)[-1] and sn[1]==sorted(FLAGS.width_mult_range)[-1] and sn[2]==sorted(FLAGS.resolution_list)[-1]:
                    #print('maxnet')
                    maxnet_output = model_student(source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=0)
                    source_ce_loss = FLAGS.gamma_ce*criterion(maxnet_output, source_target).cuda(non_blocking=True)
                    tot_source_ce_loss += source_ce_loss/FLAGS.gamma_ce
                    
                    target_rd_loss =  torch.tensor(0.0).cuda(non_blocking=True)
                    
                    maxnet_output_target = model_student(target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=1)
                    maxnet_output_target_detach = maxnet_output_target.detach()
                    pseudo_label = torch.softmax(maxnet_output_target_detach, dim=-1)
                    max_probs, targets_pl = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(FLAGS.pl_thresh).float()
                    targets_pl = torch.autograd.Variable(targets_pl)
                    # target pl loss
                    if FLAGS.use_iml:
                        info_loss = info_max_loss(torch.softmax(maxnet_output_target,dim=-1))
                        target_pl_loss = FLAGS.gamma_pl*criterion(maxnet_output_target, targets_pl).cuda(non_blocking=True) #+ info_loss.cuda(non_blocking=True)
                    else:
                        target_pl_loss = FLAGS.gamma_pl*criterion(maxnet_output_target, targets_pl).cuda(non_blocking=True) #+ info_loss #torch.tensor(0.0).cuda(non_blocking=True)
                    
                    im_pl_loss = target_pl_loss + info_loss.cuda(non_blocking=True)
                    tot_target_pl_loss += target_pl_loss/FLAGS.gamma_pl
                    
                    if FLAGS.use_dis:
                        total_loss = source_ce_loss + source_adv_loss + target_adv_loss + im_pl_loss
                    else:
                        total_loss = source_ce_loss + im_pl_loss
                    total_loss.backward()
    
                
                elif sn[0] == sorted(FLAGS.depth_mult_range)[0] and sn[1] == sorted(FLAGS.width_mult_range)[0] and sn[2] == sorted(FLAGS.resolution_list)[0]:
                    #print('minnet')
                    minnet_output = model_student(source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=0)
                    source_ce_loss = FLAGS.gamma_ce*criterion(minnet_output, source_target).cuda(non_blocking=True) #torch.tensor(0.0).cuda(non_blocking=True)
                    tot_source_ce_loss += source_ce_loss/FLAGS.gamma_ce
                    
                    minnet_output_target = model_student(target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=1)
                    
                    if FLAGS.use_iml:
                        info_loss = info_max_loss(torch.softmax(minnet_output_target,dim=-1))
                        target_rd_loss = FLAGS.gamma_rd*rd_loss(minnet_output_target, torch.mean(torch.stack(teacher_output_lst),dim=0), epoch).cuda(non_blocking=True) + info_loss.cuda(non_blocking=True)
                    else:
                        target_rd_loss = FLAGS.gamma_rd*rd_loss(minnet_output_target, torch.mean(torch.stack(teacher_output_lst),dim=0), epoch).cuda(non_blocking=True)

                    tot_rd_loss += (target_rd_loss - info_loss.cuda(non_blocking=True))/FLAGS.gamma_rd
    
                    if FLAGS.use_dis:
                        total_loss = source_ce_loss + source_adv_loss + target_adv_loss + target_rd_loss
                    else:
                        total_loss = source_ce_loss + target_rd_loss
    
                    total_loss.backward()

                else:
                    #print('random subnet')
                    subnet_out = model_student(source_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=0)
                    source_ce_loss = FLAGS.gamma_ce*criterion(subnet_out, source_target).cuda(non_blocking=True)
                    tot_source_ce_loss += source_ce_loss/FLAGS.gamma_ce
                    subnet_out_target = model_student(target_input_list[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=1)

    
                    if FLAGS.use_iml:
                        info_loss = info_max_loss(torch.softmax(subnet_out_target,dim=-1))
                        target_rd_loss = FLAGS.gamma_rd*rd_loss(subnet_out_target, teacher_output_max, epoch).cuda(non_blocking=True) + info_loss.cuda(non_blocking=True)
                    else:
                        target_rd_loss = FLAGS.gamma_rd*rd_loss(subnet_out_target, teacher_output_max, epoch).cuda(non_blocking=True)

                    
                    tot_rd_loss += (target_rd_loss - info_loss.cuda(non_blocking=True))/FLAGS.gamma_rd

                    if FLAGS.use_dis:
                        total_loss = source_ce_loss + source_adv_loss + target_adv_loss + target_rd_loss
                    else:
                        total_loss = source_ce_loss + target_rd_loss
                    total_loss.backward()
                    
                    
                
                print("Updating teachers with ema") 

                m = FLAGS.ema_decay
                for param_q, param_k in zip(model_student.parameters(), model_teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.data)
                 
                logger.info("Epoch:{}/{} Iter:{}/[s:{} t:{}] LR:{:.6f} Subnet(DxWxR):{:.1f}x{:.1f}x{} :: source_ce: {:.4f} target_rd: {:.4f} target_pl: {:.4f} info_loss: {:.4f}".format(epoch,FLAGS.num_epochs,
                                                                                                                                    batch_idx, len(source_loader), len(target_loader),
                                                                                                                                    optimizer.param_groups[0]['lr'],
                                                                                                                                    sn[0],sn[1],
                                                                                                                                 sn[2],
                                                                                                                                    (source_ce_loss.item()/FLAGS.gamma_ce), ((target_rd_loss.item()-info_loss.item())/FLAGS.gamma_rd), (target_pl_loss.item()/FLAGS.gamma_pl),info_loss.item()))
           
            optimizer.step()

            if epoch>FLAGS.warm_ep:
                lr_scheduler.step()
        except Exception as e: 
            print(e)
            continue
        


class RDLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        # teacher centering and sharpening
        temp = self.teacher_temp
        teacher_out = F.softmax((teacher_output- self.center)/ temp, dim=-1)
        teacher_out = teacher_out.detach()
        student_out = F.log_softmax(student_out, dim=-1)
             
        total_loss = 0

        loss = torch.sum((-teacher_out*student_out),dim = -1)

        return loss.mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, keepdim=True)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    
    
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def validate(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size
    with torch.no_grad():
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(lambda r: setattr(r, 'res', resolution))
            model.apply(lambda d: setattr(d, 'depth_mult', 1))
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                output = model(input[FLAGS.resolution_list.index(resolution)].cuda(non_blocking=True),dom=1)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('VAL {:.1f}s {}x Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                time.time() - t_start, str(width_mult), epoch,
                FLAGS.num_epochs, loss/cnt, acc/cnt))
    v_loss.append(loss/cnt)
    acc_lst.append(acc/cnt)
    return acc/cnt

def test(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        subnets = [[1,1,224],[1,0.9,224],[1,1,192],[1,0.9,192],[0.5,1,224],[1,1,160],[0.5,0.9,224],[1,0.9,160],[0.5,1,192],[1,1,128],[0.5,0.9,192],[1,0.9,128],[0.5,1,160],[0.5,0.9,160],[0.5,1,128],[0.5,0.9,128]]
        for sn in subnets:
            model.apply(lambda r: setattr(r, 'res', sn[2]))
            model.apply(lambda m: setattr(m, 'width_mult', sn[1]))
            model.apply(lambda d: setattr(d, 'depth_mult', sn[0]))
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                target = target.cuda(non_blocking=True)
                output = model(input[FLAGS.resolution_list.index(sn[2])].cuda(non_blocking=True),dom=1)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices==target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('VAL {:.1f}s Subnet(DxWxR):{:.1f}x{:.1f}x{} Epoch:{}/{} Loss:{:.4f} Acc:{:.1f}'.format(
                time.time() - t_start, sn[0], sn[1], sn[2], epoch,
                FLAGS.num_epochs, loss/cnt, (acc/cnt)*100))


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=31, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = FLAGS.num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # Cross Entropy loss after smoothing the labels
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)

        return loss


def train_val_test():
    """train and val"""
    global best_prec1
    # seed
    set_random_seed()
    # model
    model_student = get_model()
    model_teacher= get_model()
    model_student_wrapper = torch.nn.DataParallel(model_student).cuda()
    model_teacher_wrapper = torch.nn.DataParallel(model_teacher).cuda()

    if FLAGS.lbl_smooth:
        criterion = CrossEntropyLabelSmooth().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    
    source_loader, target_loader, val_loader = get_dataset()
    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model_student' in checkpoint:
            checkpoint_student = checkpoint['model_student']
            checkpoint_teacher = checkpoint['model_student'] ######### not teacher
        new_keys_student = list(model_student_wrapper.state_dict().keys())
        old_keys_student = list(checkpoint_student.keys())
        new_keys_teacher = list(model_teacher_wrapper.state_dict().keys())
        old_keys_teacher = list(checkpoint_teacher.keys())
        new_checkpoint_student = {}
        for key_new, key_old in zip(new_keys_student, old_keys_student):
            new_checkpoint_student[key_new] = checkpoint_student[key_old]
        model_student_wrapper.load_state_dict(new_checkpoint_student, strict=True)
        new_checkpoint_teacher = {}
        for key_new, key_old in zip(new_keys_teacher, old_keys_teacher):
            new_checkpoint_teacher[key_new] = checkpoint_teacher[key_old]
        model_teacher_wrapper.load_state_dict(new_checkpoint_teacher, strict=True)
        print('Loaded model {}.'.format(FLAGS.pretrained))
        
    optimizer = get_optimizer(model_student_wrapper)
    # check resume training
    loader_size = len(source_loader) if len(source_loader) > len(target_loader) else len(target_loader)
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        model_student_wrapper.load_state_dict(checkpoint['model_student'])
        model_teacher_wrapper.load_state_dict(checkpoint['model_student'])
        last_epoch = checkpoint['last_epoch']
        optimizer.param_groups[0]['lr']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size*FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.resume, last_epoch))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, loader_size*FLAGS.num_epochs)
        last_epoch = lr_scheduler.last_epoch
        # print model and do profiling
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling_eda(model_student, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model_student, use_cuda=False)
        print(model_student_wrapper)
    # ============ preparing loss ... ============
    rd_loss = RDLoss(
        FLAGS.num_classes,
        FLAGS.warmup_teacher_temp,
        FLAGS.teacher_temp,
        FLAGS.warmup_teacher_temp_epochs,
        FLAGS.num_epochs,
    ).cuda()
    
    
    if FLAGS.test_only:
        logger.info('Start testing.')
        print('{}eps_soep{}_lr{}_bs{}_gc{}gd{}_{}to{}.pt'.format(FLAGS.num_epochs,FLAGS.sonly_ep,FLAGS.lr,FLAGS.batch_size,FLAGS.gamma_ce,FLAGS.gamma_rd,FLAGS.sdomain, FLAGS.tdomain))
        test(last_epoch, val_loader, model_student_wrapper, criterion, source_loader)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs + 1):
        # train
        train(epoch, source_loader, target_loader, rd_loss, model_student_wrapper, model_teacher_wrapper, criterion, optimizer, lr_scheduler)
        # val
        print('Student Acc:')
        prec1 = validate(epoch, val_loader, model_student_wrapper, criterion, source_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        output_best = 'Best Prec@1: %.3f\n' % (best_prec1*100)
        print(output_best)
        if is_best:
            torch.save(
                {
                    'model_student': model_student_wrapper.state_dict(),
                    'model_teacher': model_teacher_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                },
                os.path.join(saved_path, 'waugnoddisltcosaftepcheckpoint_bestin{}eps_soep{}_lr{}_bs{}x{}_gc{}gd{}gpl{}_sd_{}_{}to{}.pt'.format(FLAGS.num_epochs,FLAGS.sonly_ep,FLAGS.lr,FLAGS.s_bs,FLAGS.t_bs,FLAGS.gamma_ce,FLAGS.gamma_rd,FLAGS.gamma_pl,FLAGS.random_seed,FLAGS.sdomain, FLAGS.tdomain)))         

    print('{}eps_soep{}_lr{}_bs{}_gc{}gd{}gpl{}_{}to{}.pt'.format(FLAGS.num_epochs,FLAGS.sonly_ep,FLAGS.lr,FLAGS.batch_size,FLAGS.gamma_ce,FLAGS.gamma_rd,FLAGS.gamma_pl,FLAGS.sdomain, FLAGS.tdomain))
    print("{} --> {}".format(FLAGS.sdomain, FLAGS.tdomain))

    test(last_epoch, val_loader, model_student_wrapper, criterion, source_loader)
    print("Teacher:")
    test(last_epoch, val_loader, model_teacher_wrapper, criterion, source_loader)
    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()