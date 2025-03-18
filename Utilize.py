# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
# from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm
import random
import sys
sys.path.insert(0, '..//Filt_Packages')
sys.path.insert(0, '..//CCA_Packages')
sys.path.insert(0, '..//Loss_Packages')
import numpy as np
import os
import math
from tqdm import tqdm

class param():
    def __init__(self, options):
        cls_num = options["cls_num"]
        src_ncls = options["src_ncls"]
        
        self.batch_size = src_ncls  # default:64, input batch size for training
        self.test_batch_size = src_ncls # default:64, input batch size for testing
        
        self.train_lr = 1e-2
        self.train_pre_epoch = 15
        self.train_final_epoch = 25
        self.train_milestone = [5, 15]
        self.train_gamma = 0.1
        
        m1 = 0.9
        n1 = 0.6
        alpha1 = 5/3 * (m1-n1)
        beta1 = n1 - 1/5 * alpha1
        
        m2 = 0.5
        n2 = 0.0
        alpha2 = 5/3 * (m2-n2)
        beta2 = n2 - 1/5 * alpha2
        
        if src_ncls / cls_num >= 0.8:
            self.zs_forcast = m1  # predict accuracy of unseen related to seen classes
            self.pred_comb = m2  # weight for transformer model
        else:
            self.zs_forcast = beta1 + alpha1 * (src_ncls / cls_num)
            self.pred_comb = beta2 + alpha2 * (src_ncls / cls_num)
            
        if self.zs_forcast < n1: self.zs_forcast = n1
        if self.pred_comb < n2: self.pred_comb = n2
        
        self.inter_channels = 8
        self.log_interval = 30  # default:10, how many batches to wait before logging training status
    
        self.no_cuda = False  # default:False, disables CUDA training
        self.dry_run = False  # default:False, quickly check a single pass
        self.save_model = True  # default:False, For Saving the current Model

class MainLite():
    def __init__(self, data, options):
        self.train_data = data["train_data"]
        self.train_label = data["train_label"]
        self.train_sine_template = data["train_sine_template"]
        
        self.vali_data = data["vali_data"]
        self.vali_label = data["vali_label"]
        self.vali_sine_template = data["vali_sine_template"]
        
        self.test_all_data = data["test_all_data"]
        self.test_all_label = data["test_all_label"]
        self.test_sine_template = data["test_sine_template"]
        
        self.f = options["freq"]
        self.phase = options["phase"]
        self.Fs = options["Fs"]
        self.harmo = options["harmo"]
        self.num_sub_band = options["num_sub_band"]
        self.cls_num = options["cls_num"]
        self.src_cls = options["src_cls"]
        self.src_ncls = options["src_ncls"]
        self.zs_ncls = options["zs_ncls"]
        self.zs_cls = options["zs_cls"]
        self.train_block_num = options["train_block_num"]
        self.num_chans = options["channel_num"]
        self.tw = int(options["tw"]*self.Fs)
        self.n_samples = options["windows"]
        self.sine_latence_window = int(options["sine_latence"]*self.Fs)
        
        self.tw_t = options["tw"]
        self.start_time = options["start_time"]
        self.start_tw = options["start_tw"]
        self.non_overlapping_window = options["non_overlapping_window"]
        
        self.xls_name = options['xls_name']
        self.xls_book = options['xls_book']
        self.xls_sheet = options['xls_sheet']
        
        self.args = param(options)
        
        torch.backends.cudnn.benchmark=True
        
    def prepareTemplate(self):
        # template: [sub_band, win, ch, cls, Ns]
        template = np.zeros([self.num_sub_band,
                             self.n_samples,
                             self.cls_num,
                             self.num_chans,
                             self.tw
                             ])
        
        for sub_band in range(self.num_sub_band):
            for cls_idx in range(self.src_ncls):
                cls_i = self.src_cls[cls_idx]
                for sample_idx in range(self.n_samples):
                    template[sub_band,
                             sample_idx,
                             cls_i,:,:] = \
                        np.squeeze(
                            np.mean(self.train_data[sample_idx+cls_idx*self.n_samples \
                                                    ::self.n_samples*self.src_ncls, \
                                                    sub_band,:,:], axis=0)
                        )
        self.template = template.transpose(0,1,3,2,4)
        
    def train(self, args, 
             model,
             device, 
             train_loader,
             optimizer,
             epoch):
        criterion_cls = nn.CrossEntropyLoss()
        model.train()
        
        batch_idx = 0
        train_loss_mat = np.zeros((1, 1))
        with tqdm(train_loader, leave=False) as tbar:
            for data, target in tbar:
                correct = 0
                x = data["x"]
                s = data["s"]
                t = data["t"]
                x, s, t, target = x.to(device), s.to(device), t.to(device), target.to(device)
                optimizer.zero_grad()
                C1, C2, C3 = model(x, s, t, batch_idx, epoch)
                
    
                C1_loss = criterion_cls(C1, target.long())
                C2_loss = criterion_cls(C2, target.long())
                C3_loss = criterion_cls(C3, target.long())
                
                loss = C1_loss + C2_loss + C3_loss

                loss.backward()
                optimizer.step()
                batch_idx += 1
                
                predsum = ((1-args.pred_comb)*C1+args.pred_comb*C2).argmax(dim=1, keepdim=True)
                pred1 = C1.argmax(dim=1, keepdim=True)
                pred2 = C2.argmax(dim=1, keepdim=True)
                acc_test = np.array(
                            [predsum.eq(target.view_as(predsum)).sum().item(),
                            pred1.eq(target.view_as(pred1)).sum().item(),
                            pred2.eq(target.view_as(pred2)).sum().item(),
                            ])
                correct = np.max(acc_test)
                
                if batch_idx % args.log_interval == 0:
                    loss_item = loss.item()
                    tbar.set_postfix({"Epoch": epoch,
                                      "Loss": loss_item,
                                      "Acc": correct / args.batch_size})
                    train_loss_mat = np.append(train_loss_mat, loss_item)
                    if args.dry_run:
                        break
        return train_loss_mat
    
    def vali(self, args, 
             model,
             device, 
             vali_loader,
             win_idx, epoch):
        # criterion_cls = nn.CrossEntropyLoss()
        correct_rec = 0
        batch_num = 0
        with torch.no_grad():
            for data, target in vali_loader:
                x = data["x"]
                s = data["s"]
                t = data["t"]
                x, s, t, target = x.to(device), s.to(device), t.to(device), target.to(device)
                model.eval()
        
                C1, C2, C3 = model(x, s, t)
                
                predsum = ((1-args.pred_comb)*C1+args.pred_comb*C2).argmax(dim=1, keepdim=True)
                pred1 = C1.argmax(dim=1, keepdim=True)
                pred2 = C2.argmax(dim=1, keepdim=True)
                
                acc_vali = [predsum.eq(target.view_as(predsum)).sum().item(),
                            pred1.eq(target.view_as(pred1)).sum().item(),
                            pred2.eq(target.view_as(pred2)).sum().item(),
                            ]
                
                correct = (acc_vali[1]*self.zs_ncls*args.zs_forcast +\
                            np.max(acc_vali)*self.src_ncls) / self.cls_num
                    
                correct_rec += correct
                batch_num += 1
            
            acc = 100. * correct_rec / (batch_num*args.batch_size)   
            
            P = acc / 100
            if P >= 1: P = 1-1e-16
            if P <= 0: P = 1e-16
            N = self.cls_num
            T = self.start_time + self.start_tw + win_idx*self.non_overlapping_window + self.tw_t
            ITR = (math.log2(N) +\
                   P*math.log2(P) +\
                   (1-P)*math.log2((1-P)/(N-1))) *\
                   60/T
                   
            print('Vali Acc:  {:.2f}'.format(acc))
                
        return acc_vali, acc, ITR
        
    def test(self, args, 
             model,
             device, 
             test_loader,
             epoch,
             net_ind = 0):
        criterion_cls = nn.CrossEntropyLoss()
        test_loss = 0
        correct_rec = 0
        zs_num = 0
        zs_correct = 0
        batch_num = 0
        with torch.no_grad():
            for data, target in test_loader:
                # start_exp = time.time()
                x = data["x"]
                t = data["t"]
                s = data["s"]
                x, t, s, target = x.to(device), t.to(device), s.to(device), target.to(device)
                model.eval()
                
                C1, C2, C3 = model(x, s, t)
                
                predsum = ((1-args.pred_comb)*C1+args.pred_comb*C2).argmax(dim=1, keepdim=True)
                pred1 = C1.argmax(dim=1, keepdim=True)
                pred2 = C2.argmax(dim=1, keepdim=True)
                pred = torch.zeros(args.batch_size, device=device)
                
                C1_loss = criterion_cls(C1, target.long())
                C2_loss = criterion_cls(C2, target.long())
                C3_loss = criterion_cls(C3, target.long())
                
                loss = C1_loss + C2_loss + C3_loss
                test_loss += loss

                pred_mat = [predsum, pred1, pred2]
                for batch_ind in range(args.batch_size):
                    if pred1[batch_ind].item() in self.zs_cls:
                        pred[batch_ind] = pred1[batch_ind]
                    else:
                        pred[batch_ind] = pred_mat[net_ind][batch_ind]
                
                batch_num += 1
                correct = pred.eq(target.view_as(pred)).sum().item()

                correct_rec += correct
        
                for batch_ind in range(args.batch_size):
                    if target[batch_ind] in self.zs_cls:
                        zs_num += 1
                        if pred[batch_ind] == target[batch_ind]:
                            zs_correct += 1

        acc = 100. * correct_rec / (batch_num*args.batch_size)
        if zs_num > 0:
            zs_acc = 100. * zs_correct/zs_num
        else:
            zs_acc = 1.0
        test_loss /= (batch_num*args.batch_size)
        print('Test epoch:{} Acc: {}/{} ({:.2f}%) ZS_Acc: {}/{} ({:.2f}%) Loss: {:.2f}\n'.format(
            epoch, 
            correct_rec, batch_num*args.batch_size, acc, 
            zs_correct, zs_num, zs_acc, 
            test_loss))
        return acc, zs_acc
        
    def trainModel(self, model_name):
        args = self.args
        torch.backends.cudnn.benchmark=True
        
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 4,  # default:1
                           'pin_memory': False,  # default:True
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        dataset = {
            "x": self.train_data.astype('float32'),
            "template": self.template.astype('float32'),
            "sine_template": self.train_sine_template.astype('float32'),
            "label": self.train_label
        }
        dataset_v = {
            "x": self.vali_data.astype('float32'),
            "template": self.template.astype('float32'),
            "sine_template": self.vali_sine_template.astype('float32'),
            "label": self.vali_label
        }
        dataset_t = {
            "x": self.test_all_data.astype('float32'),
            "template": self.template.astype('float32'),
            "sine_template": self.test_sine_template.astype('float32'),
            "label": self.test_all_label
        }
        train_loader = torch.utils.data.DataLoader(TrainDatasets(dataset), drop_last=True, **train_kwargs)
        
        train_loader_D_list = []
        vali_loader_D_list = []
        test_all_loader_D_list = []
        for win_idx in range(self.n_samples):
            train_loader_D_list.append(torch.utils.data.DataLoader(TrainDatasets_D(dataset, win_idx), drop_last=True, **train_kwargs))
            vali_loader_D_list.append(torch.utils.data.DataLoader(TestDatasets_D(dataset_v, win_idx), drop_last=True, **test_kwargs))
            test_all_loader_D_list.append(torch.utils.data.DataLoader(TestDatasets_D(dataset_t, win_idx), drop_last=True, **test_kwargs))
        
        params = {
            "tw": self.tw,
            "Fs": self.Fs,
            "ch": self.num_chans,
            "cls_num": self.cls_num,
            "freq": self.f,
            "harmo": self.harmo,
            "num_sub_band": self.num_sub_band,
            "sine_latence_window": self.sine_latence_window,
            "train_block_num": self.train_block_num,
            "batch_size": args.batch_size,
            "inter_ch": args.inter_channels,
            "device": device,
        }
        model = GLite(params).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                                lr=args.train_lr, 
                                betas=(0.9,0.999),
                                eps=1e-8,
                                weight_decay=1e-3)
        ms_scheduler = MultiStepLR(optimizer, milestones=args.train_milestone, gamma=args.train_gamma)
        
        for epoch in range(1, args.train_pre_epoch + 1):
            train_loss = self.train(args, 
                                    model,
                                    device, 
                                    train_loader, 
                                    optimizer,
                                    epoch)
            train_loss = train_loss[1:]
            data_len = len(train_loss)
            
            ms_scheduler.step()
            
            acc, zs_acc = self.test(args, 
                            model, 
                            device, 
                            test_all_loader_D_list[3],
                            epoch)
            
            self.xls_sheet.write(epoch, 0, epoch)
            for idx in range(data_len):
                self.xls_sheet.write(epoch, idx+1, train_loss[idx])
            self.xls_sheet.write(epoch, data_len+2, acc)
            self.xls_sheet.write(epoch, data_len+3, zs_acc)
            self.xls_book.save(filename_or_stream=self.xls_name)
        
        win_ITR_rec = np.zeros(self.n_samples)
        for win_idx in range(self.n_samples):
            #############################################
            if win_idx*self.non_overlapping_window +\
                self.start_tw  > 1.6:
                break
            #############################################
            
            acc_vali_list, acc, ITR_vali = self.vali(args, 
                                    model, 
                                    device, 
                                    vali_loader_D_list[win_idx],
                                    win_idx, epoch)
            
            vali_ITR_rec = ITR_vali
            self.xls_sheet.write(epoch, data_len+2+win_idx, acc)
            self.xls_book.save(filename_or_stream=self.xls_name)
            
            print("Vali ITR: ", vali_ITR_rec)
            win_ITR_rec[win_idx] = vali_ITR_rec

        best_win = np.argmax(win_ITR_rec)
        ft_loader = train_loader_D_list[best_win]
        va_loader = vali_loader_D_list[best_win]
        te_all_loader = test_all_loader_D_list[best_win]
        test_acc_rec = np.zeros((args.train_final_epoch+1))
        
        for epoch in range(args.train_pre_epoch + 1, args.train_final_epoch+1):  
            tab_len = 4
            train_loss = self.train(args, 
                                        model,
                                        device, 
                                        ft_loader, 
                                        optimizer,
                                        epoch)
            ms_scheduler.step()
            train_loss = train_loss[1:]
            data_len = len(train_loss)
            self.xls_sheet.write(epoch, 0, epoch)
            for idx in range(data_len):
                self.xls_sheet.write(epoch, idx+1, train_loss[idx])
            self.xls_book.save(filename_or_stream=self.xls_name)
            
            acc_vali_list, acc_vali, ITR_vali = self.vali(args, 
                                    model, 
                                    device, 
                                    va_loader,
                                    best_win, epoch)
            net_ind = acc_vali_list.index(np.max(acc_vali_list))
                
            acc, zs_acc = self.test(args, 
                            model, 
                            device, 
                            te_all_loader,
                            epoch,
                            net_ind = net_ind)
            
            test_acc_rec[epoch] = acc
            
            p = acc / 100
            if p >= 1: p = 1-1e-16
            if p <= 0: p = 1e-16
            n = self.cls_num
            t = self.start_time + self.start_tw + best_win*self.non_overlapping_window + self.tw_t
            itr = (math.log2(n) +\
                   p*math.log2(p) +\
                   (1-p)*math.log2((1-p)/(n-1))) *\
                   60/t
                   
            self.xls_sheet.write(epoch, tab_len+2, acc)
            self.xls_sheet.write(epoch, tab_len+3, itr)
            self.xls_sheet.write(epoch, tab_len+4, acc_vali)
            self.xls_sheet.write(epoch, tab_len+5, ITR_vali)
            self.xls_book.save(filename_or_stream=self.xls_name)
        
        return test_acc_rec, best_win
        
class TrainDatasets(Dataset):
    def __init__(self, inargs):
        super().__init__()
        self.x = inargs["x"]
        self.t = inargs["template"]
        self.sine_template = inargs["sine_template"]
        self.label = inargs["label"]
        
        n_samples = self.sine_template.shape[0]
        
        self.src, self.trg = [], []
        for idx in range(self.x.shape[0]):
            t = self.t[:,idx%n_samples,:,:,:]
            s = self.sine_template[idx%n_samples,:,:,:]
            self.src.append({
                "x": self.x[idx,:,:,:],
                "t": t,
                "s": s
                })
            self.trg.append(self.label[idx])
           
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src) 
    
class TestDatasets(Dataset):
    def __init__(self, inargs):
        super().__init__()
        self.x = inargs["x"]
        self.t = inargs["template"]
        self.sine_template = inargs["sine_template"]
        self.label = inargs["label"]
        
        n_samples = self.sine_template.shape[0]
        
        self.src, self.trg = [], []
        for idx in range(self.x.shape[0]):
            t = self.t[:,idx%n_samples,:,:,:]
            s = self.sine_template[idx%n_samples,:,:,:]
            self.src.append({
                "x": self.x[idx,:,:,:],
                "t": t,
                "s": s
                })
            self.trg.append(self.label[idx])
            
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)  
    
class TrainDatasets_D(Dataset):
    def __init__(self, inargs, win_idx):
        super().__init__()
        self.x = inargs["x"]
        self.t = inargs["template"]
        self.sine_template = inargs["sine_template"]
        self.label = inargs["label"]
        
        n_samples = self.sine_template.shape[0]
        
        self.src, self.trg = [], []
        for idx in range(win_idx, self.x.shape[0], n_samples):
            t = self.t[:,idx%n_samples,:,:,:]
            s = self.sine_template[idx%n_samples,:,:,:]
            self.src.append({
                "x": self.x[idx,:,:,:],
                "t": t,
                "s": s
                })
            self.trg.append(self.label[idx])
            
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src) 
    
    
class TestDatasets_D(Dataset):
    def __init__(self, inargs, win_idx):
        super().__init__()
        self.x = inargs["x"]
        self.t = inargs["template"]
        self.sine_template = inargs["sine_template"]
        self.label = inargs["label"]
        
        n_samples = self.sine_template.shape[0]
        
        self.src, self.trg = [], []
        for idx in range(win_idx, self.x.shape[0], n_samples):
            t = self.t[:,idx%n_samples,:,:,:]
            s = self.sine_template[idx%n_samples,:,:,:]
            self.src.append({
                "x": self.x[idx,:,:,:],
                "t": t,
                "s": s
                })
            self.trg.append(self.label[idx])
            
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src) 

class GLite(nn.Module):
    def __init__(self, params):
        super(GLite, self).__init__()
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.inter_channels = params["inter_ch"]
        self.num_sub_band = params['num_sub_band']
        self.freq = params['freq']
        self.harmo = params["harmo"]
        self.sine_latence_window = params['sine_latence_window']
        self.cls_num = params['cls_num']
        self.tr_num = params['train_block_num']
        self.batch_size = params["batch_size"]
        self.device = params["device"]
        
        self.SB_ES = nn.ModuleList()
        self.SB_ET = nn.ModuleList()
        self.SB_EK = nn.ModuleList()
        self.SB_EV = nn.ModuleList()
        
        self.SB_WS1 = nn.ModuleList()
        self.SB_WT1 = nn.ModuleList()
        self.SB_WK = nn.ModuleList()
        self.SB_WV = nn.ModuleList()
        
        self.SB_WS2 = nn.ModuleList()
        self.SB_WT2 = nn.ModuleList()
        self.SB_WSX = nn.ModuleList()
        self.SB_WTX = nn.ModuleList()
        
        self.SB_ST = nn.ModuleList()
        
        self.SB_WOsx = nn.ModuleList()
        self.SB_WOtx = nn.ModuleList()
        
        self.U1 = SubmLayer(self.num_sub_band)
        self.U2 = SubmLayer(self.num_sub_band)
        self.U3 = SubmLayer(self.num_sub_band)

        self.SB_ES = SineEmbedding(params)
        self.SB_ET = TemplateEmbedding(params)
        self.SB_EK = KxEmbedding(params)
        self.SB_EV = VxEmbedding(params)
        
        self.SB_WS1 = ProjLayer(self.inter_channels, self.cls_num)
        self.SB_WT1 = ProjLayer(self.inter_channels, self.cls_num)
        self.SB_WK = ProjLayer(self.inter_channels, 1)
        self.SB_WV = ProjLayer(self.inter_channels, 1)
        
        self.SB_WS2 = ProjLayer(self.inter_channels, self.cls_num)
        self.SB_WT2 = ProjLayer(self.inter_channels, self.cls_num)
        self.SB_WSX = ProjLayer(self.inter_channels, self.cls_num)
        self.SB_WTX = ProjLayer(self.inter_channels, self.cls_num)
        
        self.SB_ST = STLayer(self.inter_channels, self.cls_num)
        
        self.SB_WOsx = OutpLayer(self.inter_channels)
        self.SB_WOtx = OutpLayer(self.inter_channels)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, X, S, T, batch_idx=0, epoch=0):
        C1 = torch.empty((self.batch_size, 
                          self.num_sub_band, 
                          self.cls_num), 
                          device=self.device)
        C2 = torch.empty((self.batch_size, 
                          self.num_sub_band, 
                          self.cls_num), 
                          device=self.device)
        C3 = torch.empty((self.batch_size, 
                          self.num_sub_band, 
                          self.cls_num), 
                          device=self.device)
            
        x = X[:, 0, :, :].unsqueeze(2)
        t = T[:, 0, :, :, :]
        s = S[:, :, :, :]
        
        E_S = self.SB_ES(s, batch_idx, epoch)
        E_S = E_S[:, :, :, self.sine_latence_window:]
        E_T = self.SB_ET(t, batch_idx, epoch)
        E_K = self.SB_EK(x, batch_idx, epoch)
        E_V = self.SB_EV(x, batch_idx, epoch)
        
        Q_S1 = self.SB_WS1(E_S)
        Q_T1 = self.SB_WT1(E_T)
        K = self.SB_WK(E_K)
        V = self.SB_WV(E_V)
        
        Que_SX1 = F.softmax(
                (torch.matmul(Q_S1.transpose(1,2), 
                              K.transpose(1,2).transpose(2,3))) / \
                (self.inter_channels ** (1 / 2)), dim=-1).transpose(1,2)
        Att_SX1 = torch.matmul(Que_SX1,V.transpose(1,2))
        Que_TX1 = F.softmax(
                (torch.matmul(Q_T1.transpose(1,2), 
                              K.transpose(1,2).transpose(2,3))) / \
                (self.inter_channels ** (1 / 2)), dim=-1).transpose(1,2)
        Att_TX1 = torch.matmul(Que_TX1, V.transpose(1,2))
        
        Q_S2 = self.SB_WS2(E_S)
        Q_T2 = self.SB_WT2(E_T)
        SX = self.SB_WSX(Att_SX1) 
        TX = self.SB_WTX(Att_TX1) 
        
        Que_SX2 = F.softmax(
                (torch.matmul(SX.transpose(1,2), 
                              Q_S2.transpose(1,2).transpose(2,3))) / \
                (self.inter_channels ** (1 / 2)), dim=-1).transpose(1,2)
        Que_TX2 = F.softmax(
                (torch.matmul(TX.transpose(1,2), 
                              Q_T2.transpose(1,2).transpose(2,3))) / \
                (self.inter_channels ** (1 / 2)), dim=-1).transpose(1,2)
            
        C1[:,0,:] = self.SB_WOsx(Que_SX2).squeeze()
        C2[:,0,:] = self.SB_WOtx(Que_TX2).squeeze()
        C3[:,0,:] = self.SB_ST(E_S, E_T).squeeze()

        if self.num_sub_band > 1:
            C1 = self.U1(C1.transpose(1,2))
            C2 = self.U2(C2.transpose(1,2))
            C3 = self.U3(C3.transpose(1,2))
        return C1.squeeze(), C2.squeeze(), C3.squeeze()

class SineEmbedding(nn.Module):
    def __init__(self, params):
        super(SineEmbedding, self).__init__()
        self.Fs = params['Fs']
        self.inter_channels = params["inter_ch"]
        self.num_sub_band = params['num_sub_band']
        self.freq = params['freq']
        self.harmo = params["harmo"]
        self.sine_latence_window = params['sine_latence_window']
        self.cls_num = params['cls_num']
        self.tr_num = params['train_block_num']
        self.batch_size = params["batch_size"]
        self.la = 1e-1
        self.device = params["device"]
        
        self.norm = nn.InstanceNorm2d(self.cls_num,affine=False)
        
        hidden_channels = [self.inter_channels]*4
        self.embedding = TemporalConvNet(self.harmo*2, hidden_channels, kernel_size=3)
        
        self.norm_out = nn.InstanceNorm2d(self.cls_num,affine=False)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, S, batch_idx=0, epoch=0):
        S = self.norm(S.transpose(1,2)).transpose(1,2)
        S = self.embedding(S)
        out = self.norm_out(S.transpose(1,2)).transpose(1,2)
        return out
    
class TemplateEmbedding(nn.Module):
    def __init__(self, params):
        super(TemplateEmbedding, self).__init__()
        self.Fs = params['Fs']
        self.inter_channels = params["inter_ch"]
        self.num_sub_band = params['num_sub_band']
        self.freq = params['freq']
        self.harmo = params["harmo"]
        self.sine_latence_window = params['sine_latence_window']
        self.cls_num = params['cls_num']
        self.tr_num = params['train_block_num']
        self.batch_size = params["batch_size"]
        self.la = 1e-1
        self.device = params["device"]
        
        self.norm = nn.InstanceNorm2d(self.cls_num,affine=False)
        self.embedding = nn.Sequential(
            nn.Conv2d(params["ch"], self.inter_channels,
                (1,3), 1, padding="same"),
            )
        self.norm_out = nn.InstanceNorm2d(self.cls_num,affine=False)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, T, batch_idx=0, epoch=0):
        T = self.norm(T.transpose(1,2)).transpose(1,2)
        T = self.embedding(T)
        out = self.norm_out(T.transpose(1,2)).transpose(1,2)
        return out
    
class KxEmbedding(nn.Module):
    def __init__(self, params):
        super(KxEmbedding, self).__init__()
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.inter_channels = params["inter_ch"]
        self.num_sub_band = params['num_sub_band']
        self.freq = params['freq']
        self.harmo = params["harmo"]
        self.sine_latence_window = params['sine_latence_window']
        self.cls_num = params['cls_num']
        self.tr_num = params['train_block_num']
        self.batch_size = params["batch_size"]
        self.device = params["device"]
        
        self.norm = nn.InstanceNorm2d(1,affine=False)
        self.embedding = nn.Sequential(
            nn.Conv2d(params["ch"], self.inter_channels, 
                      (1,1), 1, padding="same"),
            )
        self.norm_out = nn.InstanceNorm2d(1,affine=False)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, X, batch_idx=0, epoch=0):
        X = self.norm(X.transpose(1,2)).transpose(1,2)
        X = self.embedding(X)
        out = self.norm_out(X.transpose(1,2)).transpose(1,2)
        return out
    
class VxEmbedding(nn.Module):
    def __init__(self, params):
        super(VxEmbedding, self).__init__()
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.inter_channels = params["inter_ch"]
        self.num_sub_band = params['num_sub_band']
        self.freq = params['freq']
        self.harmo = params["harmo"]
        self.sine_latence_window = params['sine_latence_window']
        self.cls_num = params['cls_num']
        self.tr_num = params['train_block_num']
        self.batch_size = params["batch_size"]
        self.device = params["device"]
        
        self.norm = nn.InstanceNorm2d(1,affine=False)
        self.embedding = nn.Sequential(
            nn.Conv2d(params["ch"], self.inter_channels, 
                      (1,1), 1, padding="same"),
            )
        self.norm_out = nn.InstanceNorm2d(1,affine=False)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, X, batch_idx=0, epoch=0):
        X = self.norm(X.transpose(1,2)).transpose(1,2)
        X = self.embedding(X)
        out = self.norm_out(X.transpose(1,2)).transpose(1,2)
        return out
    
class ProjLayer(nn.Module):
    def __init__(self, emb_size, cls_num):
        # emb_size = [inter_ch, Nf, Ns]
        super(ProjLayer, self).__init__()
        self.projection = nn.Conv2d(emb_size, emb_size,
                      1, 1, padding="valid")
        self.norm_out = nn.InstanceNorm2d(cls_num, affine=False)
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, data):
        # data = self.projection(data)
        # out = self.norm_out(data.transpose(1,2)).transpose(1,2)
        # return out
    
        # out = self.norm_out(data.transpose(1,2)).transpose(1,2)
       
        return data
    
class STLayer(nn.Module):
        def __init__(self, emb_size, cls_num):
            super(STLayer, self).__init__()
            self.projectionS = nn.Conv2d(emb_size, 1,
                          1, 1, padding="valid")
            self.norm_S = nn.InstanceNorm2d(1,affine=False)
            
            self.projectionT = nn.Conv2d(emb_size, 1,
                          1, 1, padding="valid")
            self.norm_T = nn.InstanceNorm2d(1,affine=False)
        
        def _init_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    nn.init.zeros_(m.bias.data)
                    
        def forward(self, S, T):
            S = self.norm_S(self.projectionS(S))
            T = self.norm_T(self.projectionT(T))
            
            out = F.softmax(
                    (torch.matmul(S.transpose(1,2), 
                                  T.transpose(1,2).transpose(2,3))) / \
                    (1 ** (1 / 2)), dim=-1).transpose(1,2)
                
            return out
    
class OutpLayer(nn.Module):
    def __init__(self, emb_size):
        super(OutpLayer, self).__init__()
        self.output = nn.Sequential(
                nn.Conv2d(emb_size, 1, 
                          1, 1, padding="valid"),
                nn.Conv2d(1, 1, 
                          (1, emb_size), 1, padding="valid"),
                )
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, data):
        return self.output(data)
    
class SubmLayer(nn.Module):
    def __init__(self, num_sub_band):
        super(SubmLayer, self).__init__()
        self.output = nn.Linear(num_sub_band, 1)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, data):
        return self.output(data)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1,kernel_size),
                                           stride=(1,stride), padding=(0,padding), dilation=(1,dilation)))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1,kernel_size),
                                           stride=(1,stride), padding=(0,padding), dilation=(1,dilation)))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.conv3 = nn.Conv2d(n_outputs*2, n_outputs, 1)
        self.relu = nn.ReLU()
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        output = self.conv3(torch.cat([out, res], dim=1))
        return output


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)