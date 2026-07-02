import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import exp, log
import torch.nn.functional as F
from torch.autograd import Variable

def process_batch(data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL, batch_classify_dim=2, condition_dims=[2, 2], epoch=1, weight_modality=[1,1], nfeatures=[500,192]):
    data = Variable(data_label['data']).to(device)
    data = torch.reshape(data, (data.size(0), -1))
    mask_recon = Variable(data_label['mask_recon']).to(device)
    mask_recon = torch.reshape(mask_recon, (data.size(0), -1))
    mask_poe = Variable(data_label['mask_poe']).to(device)
    mask_poe = torch.reshape(mask_poe, (data.size(0), -1))
    label = Variable(data_label['label']).to(device)
    batch = Variable(data_label['batch']).to(device)
    condition_list = [Variable(c).to(device) for c in data_label['conditions']]

    # One-hot encode batch
    b = batch.long()
    b_onehot = F.one_hot(b, num_classes=batch_classify_dim).float()
    criterion_mse = nn.MSELoss(reduction='none')
    x, x_batch, z_embedding, z_batch, z_conditions, cla_cty1, cla_batch1, cla_conditions, mu, log_var = generator(data, batch, mask_poe)

    cty_loss1 = criterion_smooth_cty(cla_cty1, label)
    batch_loss1 = criterion_smooth_cty(cla_batch1, batch)
    kl_loss = criterion_KL(mu, log_var)

    start = 0
    ae_loss_modality = []
    for weight, feat_dim in zip(weight_modality, nfeatures):
        end = start + feat_dim
        data_slice = data[:, start:end]
        x_slice = x[:, start:end]
        temp_mse = weight * criterion_mse(data_slice, x_slice)
        ae_loss_modality.append(temp_mse)
        start = end
    ae_loss_modality = torch.cat(ae_loss_modality, dim=1)
    ae_loss1 = (ae_loss_modality * mask_recon).sum() / (mask_recon.sum() + 1e-8)
    ae_loss2 = criterion(b_onehot, x_batch)
    con_losses = []
    for i, cond in enumerate(condition_list):
        con_loss = criterion_smooth_cty(cla_conditions[i], cond).mean()
        con_losses.append(con_loss)
    return cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_losses, kl_loss


def process_batch_with_query(data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                             batch_classify_dim=2, condition_dims=[2, 2], epoch=1,
                             weight_modality=[1, 1], nfeatures=[500, 192]):

    data = Variable(data_label['data']).to(device)
    data = torch.reshape(data, (data.size(0), -1))
    mask_recon = Variable(data_label['mask_recon']).to(device)
    mask_recon = torch.reshape(mask_recon, (data.size(0), -1))
    mask_poe = Variable(data_label['mask_poe']).to(device)
    mask_poe = torch.reshape(mask_poe, (data.size(0), -1))
    label = Variable(data_label['label']).to(device)
    batch = Variable(data_label['batch']).to(device)
    condition_list = [Variable(c).to(device) for c in data_label['conditions']]
    all_train_query_info = Variable(data_label['all_train_query_info']).to(device).float()   

    b = batch.long()
    b_onehot = F.one_hot(b, num_classes=batch_classify_dim).float()
    criterion_mse = nn.MSELoss(reduction='none')
    x, x_batch, z_embedding, z_batch, z_conditions, cla_cty1, cla_batch1, cla_conditions, mu, log_var = generator(data, batch, mask_poe)

    cty_loss1 = criterion_smooth_cty(cla_cty1, label)
    batch_loss1 = criterion_smooth_cty(cla_batch1, batch)
    kl_loss = criterion_KL(mu, log_var)

    start = 0
    ae_loss_modality = []
    for weight, feat_dim in zip(weight_modality, nfeatures):
        end = start + feat_dim
        data_slice = data[:, start:end]
        x_slice = x[:, start:end]
        temp_mse = weight * criterion_mse(data_slice, x_slice)
        ae_loss_modality.append(temp_mse)
        start = end
    ae_loss_modality = torch.cat(ae_loss_modality, dim=1)
    ae_loss1 = (ae_loss_modality * mask_recon).sum() / (mask_recon.sum() + 1e-8)
    ae_loss2 = criterion(b_onehot, x_batch)

    con_losses = []
    for i, cond in enumerate(condition_list):
        ce = F.cross_entropy(cla_conditions[i], cond, reduction='none')   
        con_loss = (ce * all_train_query_info).sum() / (all_train_query_info.sum() + 1e-8)              
        con_losses.append(con_loss)
    return cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_losses, kl_loss


def process_batch_wo_condition(data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL, batch_classify_dim=2, condition_dims=[2, 2], epoch=1, weight_modality=[1,1], nfeatures=[500,192]):
    data = Variable(data_label['data']).to(device)
    data = torch.reshape(data, (data.size(0), -1))
    mask_recon = Variable(data_label['mask_recon']).to(device)
    mask_recon = torch.reshape(mask_recon, (data.size(0), -1))
    mask_poe = Variable(data_label['mask_poe']).to(device)
    mask_poe = torch.reshape(mask_poe, (data.size(0), -1))
    label = Variable(data_label['label']).to(device)
    batch = Variable(data_label['batch']).to(device)
    condition_list = [Variable(c).to(device) for c in data_label['conditions']]

    # One-hot encode batch
    b = batch.long()
    b_onehot = F.one_hot(b, num_classes=batch_classify_dim).float()
    criterion_mse = nn.MSELoss(reduction='none')
    x, x_batch, z_embedding, z_batch, z_conditions, cla_cty1, cla_batch1, cla_conditions, mu, log_var = generator(data, batch, mask_poe)
    
    batch_loss1 = criterion_smooth_cty(cla_batch1, batch)
    kl_loss = criterion_KL(mu, log_var)

    start = 0
    ae_loss_modality = []
    for weight, feat_dim in zip(weight_modality, nfeatures):
        end = start + feat_dim
        data_slice = data[:, start:end]
        x_slice = x[:, start:end]
        temp_mse = weight * criterion_mse(data_slice, x_slice)
        ae_loss_modality.append(temp_mse)
        start = end
    ae_loss_modality = torch.cat(ae_loss_modality, dim=1)
    ae_loss1 = (ae_loss_modality * mask_recon).sum() / (mask_recon.sum() + 1e-8)
    ae_loss2 = criterion(b_onehot, x_batch)
    cty_loss1 = 0 
    con_losses = 0 
    return cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_losses, kl_loss

def set_requires_grad(generator, trainable_names):
    all_names = [
        'cty1_classify',
        'batch1_classify',
        'condition_classifiers',
        'encoder',
        'decoder',
        'batch_encoder',
        'batch_decoder'
    ]
    for name in all_names:
        module = getattr(generator, name)
        requires_grad = name in trainable_names
        for param in module.parameters():
            param.requires_grad = requires_grad

def set_requires_grad_wo_condition(generator, trainable_names):
    all_names = [
        'batch1_classify',
        'encoder',
        'decoder',
        'batch_encoder',
        'batch_decoder'
    ]
    for name in all_names:
        module = getattr(generator, name)
        requires_grad = name in trainable_names
        for param in module.parameters():
            param.requires_grad = requires_grad
            
def evaluate_validation_loss(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, batch_classify_dim, condition_dim, epoch, weight_modality = [1,1], nfeatures = [500,192]):
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_sample in val_dl:
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch(
                batch_sample, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
            total_loss = ae_loss1 + ae_loss2 
            val_loss += total_loss.item()
    return val_loss / len(val_dl)

def evaluate_validation_loss_with_query(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, batch_classify_dim, condition_dim, epoch, weight_modality = [1,1], nfeatures = [500,192]):
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_sample in val_dl:
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_with_query(
                batch_sample, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
            total_loss = ae_loss1 + ae_loss2 
            val_loss += total_loss.item()
    return val_loss / len(val_dl)
    
def evaluate_validation_loss_wo_condition(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, batch_classify_dim, condition_dim, epoch, weight_modality = [1,1], nfeatures = [500,192]):
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_sample in val_dl:
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_wo_condition(
                batch_sample, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
            total_loss = ae_loss1 + ae_loss2 
            val_loss += total_loss.item()
    return val_loss / len(val_dl)
    
def train_M3(dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, lr=0.01, num_epochs=200, batch_classify_dim=2,condition_dim=2, min_delta = 0.001, early_stop_patience = 200, weight_batch_ae = 1, weight_modality = [1,1], nfeatures = [500,192]):
    optimizer_generator = torch.optim.AdamW([{'params': generator.parameters()}], lr=lr, weight_decay=1e-2)
    best_val_loss = float('inf')
    best_model_state = None
    best_model_path = 'best_model.pth'
    min_delta = min_delta

    for epoch in tqdm(range(1, num_epochs + 1)):
        batch_weight = max(1, 40.0*(1-epoch/50))
        generator.train()
        
        for batch_idx, batch_sample in enumerate(dl):
            data_label = batch_sample
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch(
                data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures
                )
            set_requires_grad(generator, ['cty1_classify', 'condition_classifiers', 'encoder',
                                          'decoder', 'batch_encoder', 'batch_decoder'])
            total_loss = ae_loss1 + weight_batch_ae*ae_loss2 + 0.0001*kl_loss + sum(con_loss_list) - batch_weight*batch_loss1
            #print("ae_loss1",ae_loss1, "weight_batch_ae*ae_loss2",weight_batch_ae*ae_loss2, "batch_weight*batch_loss1", batch_weight*batch_loss1)
            optimizer_generator.zero_grad()
            total_loss.backward()
            optimizer_generator.step()
            
        for iteration in range(3):
            for batch_idx, batch_sample in enumerate(dl):
                data_label = batch_sample
                cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch(
                    data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                    batch_classify_dim, condition_dim, 
                    epoch, weight_modality, nfeatures
                )
                set_requires_grad(generator, ['batch1_classify'])    
                total_loss = batch_weight*batch_loss1
                optimizer_generator.zero_grad()
                total_loss.backward()
                optimizer_generator.step()
    
        # use the validation to have a test
        val_loss = evaluate_validation_loss(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device,
                                        batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = generator.state_dict()
            torch.save(generator.state_dict(), best_model_path) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                generator.load_state_dict(best_model_state)
                return generator
                
    return generator


def train_M3_with_query(dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, lr=0.01, num_epochs=200, batch_classify_dim=2,condition_dim=2, min_delta = 0.001, early_stop_patience = 200, weight_batch_ae = 1, weight_modality = [1,1], nfeatures = [500,192]):
    optimizer_generator = torch.optim.AdamW([{'params': generator.parameters()}], lr=lr, weight_decay=1e-2)
    best_val_loss = float('inf')
    best_model_state = None
    best_model_path = 'best_model.pth'
    min_delta = min_delta

    for epoch in tqdm(range(1, num_epochs + 1)):
        batch_weight = max(1, 40.0*(1-epoch/50))
        generator.train()

        for batch_idx, batch_sample in enumerate(dl):
            data_label = batch_sample
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_with_query(
                data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures
                )
            set_requires_grad(generator, ['cty1_classify', 'condition_classifiers', 'encoder',
                                          'decoder', 'batch_encoder', 'batch_decoder'])
            
            total_loss = ae_loss1 + weight_batch_ae*ae_loss2 + 0.0001*kl_loss + sum(con_loss_list) - batch_weight*batch_loss1
            #print(total_loss, "!!")
            optimizer_generator.zero_grad()
            total_loss.backward()
            optimizer_generator.step()

        generator.train()
        for iteration in range(3):
            for batch_idx, batch_sample in enumerate(dl):
                data_label = batch_sample
                cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_with_query(
                    data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                    batch_classify_dim, condition_dim, 
                    epoch, weight_modality, nfeatures
                )
                set_requires_grad(generator, ['batch1_classify'])    
                total_loss = batch_weight*batch_loss1
                optimizer_generator.zero_grad()
                total_loss.backward()
                optimizer_generator.step()
    
        # use the validation to have a test
        val_loss = evaluate_validation_loss_with_query(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device,
                                        batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = generator.state_dict()
            torch.save(generator.state_dict(), best_model_path) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                generator.load_state_dict(best_model_state)
                return generator
                
    return generator


def train_M3_wo_condition(dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device, lr=0.01, num_epochs=200, batch_classify_dim=2,condition_dim=2, min_delta = 0.001, early_stop_patience = 200, weight_batch_ae = 1, weight_modality = [1,1], nfeatures = [500,192]):
    optimizer_generator = torch.optim.AdamW([{'params': generator.parameters()}], lr=lr, weight_decay=1e-2)
    best_val_loss = float('inf')
    best_model_state = None
    best_model_path = 'best_model.pth'
    min_delta = min_delta

    for epoch in tqdm(range(1, num_epochs + 1)):
        batch_weight = max(1, 40.0*(1-epoch/50))
        generator.train()
        
        for batch_idx, batch_sample in enumerate(dl):
            data_label = batch_sample
            cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_wo_condition(
                data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures
                )
            set_requires_grad_wo_condition(generator, [ 'encoder',
                                          'decoder', 'batch_encoder', 'batch_decoder'])
            total_loss = ae_loss1 + weight_batch_ae*ae_loss2 + 0.001*kl_loss - batch_weight*batch_loss1
            optimizer_generator.zero_grad()
            total_loss.backward()
            optimizer_generator.step()
            
        for iteration in range(3):
            for batch_idx, batch_sample in enumerate(dl):
                data_label = batch_sample
                cty_loss1, batch_loss1, ae_loss1, ae_loss2, con_loss_list, kl_loss = process_batch_wo_condition(
                    data_label, device, generator, criterion_smooth_cty, criterion, criterion_KL,
                    batch_classify_dim, condition_dim, 
                    epoch, weight_modality, nfeatures
                )
                set_requires_grad_wo_condition(generator, ['batch1_classify'])    
                total_loss = batch_weight*batch_loss1
                optimizer_generator.zero_grad()
                total_loss.backward()
                optimizer_generator.step()
    
        # use the validation to have a test
        val_loss = evaluate_validation_loss_wo_condition(val_dl, generator, criterion_smooth_cty, criterion, criterion_KL, device,
                                        batch_classify_dim, condition_dim, epoch, weight_modality, nfeatures)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = generator.state_dict()
            torch.save(generator.state_dict(), best_model_path) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                generator.load_state_dict(best_model_state)
                return generator
                
    return generator

