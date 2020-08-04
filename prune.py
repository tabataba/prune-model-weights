import numpy as np
import torch


def correct(test_loader,net):
    #calculates and outputs accuracy
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.view(images.shape[0], -1)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %d %%' % (100 * correct / total))

def prune_diag(net,masks):
    #outputs how many weights have been pruned, true = pruned
    
    print('name     false  true  total')
    for n, w in net.named_parameters():
        em = masks[n]     
        isf = np.sum(~em.data.numpy())
        ist = np.sum(em.data.numpy())
        
        print(n,isf,ist,isf+ist)
        
    
def get_other_weight(n,net):
    
    #for layer with weights n (bias or weight), outputs other weights (weight or bias)
    if n.endswith('bias'):
        other_name= n.split('.')[0]+'.weight'
        
    if n.endswith('weight'):
        other_name= n.split('.')[0]+'.bias'
        
    other_weight=net.state_dict()[other_name]
    
    return other_weight.clone()


def get_weights(n,net):
    #outputs edge weights (weight) and unit weights (bias) for layer n
    
    if n.endswith('weight'):
        edge_weight = net.state_dict()[n].clone()
        unit_weight = get_other_weight(n,net)
    if n.endswith('bias'):
        edge_weight = get_other_weight(n,net)
        unit_weight = net.state_dict()[n].clone()
    return edge_weight, unit_weight


def get_fout(n,net,function):
    #returns output of a function to calculate the cut-off threshold for weights to be pruned
    edge_weight, unit_weight = get_weights(n,net)
    return function(edge_weight,unit_weight)


def get_fouts(net,name,function,locked_masks):
    # returns output of threshold function and corresponding locked mask for all layers
    
    fouts = []
    mask_outs = []
    for n, w in net.named_parameters():
        if n.endswith(name):
            mask = locked_masks[n]
            mask_outs.append(mask)
            
            edge_weight, unit_weight = get_weights(n,net)
            fout = function(edge_weight,unit_weight)
            fouts.append(fout)

    return fouts, mask_outs


def prune_step(net,locked_masks):
    for n, w in net.named_parameters():  
        if w.grad is not None and n in locked_masks: 
            w.grad[locked_masks[n]] = 0
    return net
            
    
def prune(net, locked_masks, prune_random=False, prune_weight=True, prune_bias=False, ratio=None,
          threshold=None, threshold_bias=None, function=None, function_bias=None, prune_across_layers=True):
    #prunes a neural network (i.e. zeros weights and biases and keeps them at zero) according to different thresholding rules.
    #
        #prune_random: weights are pruned randomly according to ratio.
        #prune_weight: prunes edge weights
        #prune_bias: prunes unit weights
        #prune_accross_layers: True : threshold is calculated over all layers 
        #                      False: threshold is calculated over each layer separately
        
    
    
    #set pruning ratio if none given
    if ratio is None: 
        ratio = 0.25
    
    if prune_random:
        for n, w in net.named_parameters():
            if prune_weight and n.endswith('weight'):
                
                #update mask to set a number of False values to True according to ratio
                mask = locked_masks[n]
                
                mask[~mask] = torch.from_numpy(
                    np.random.choice(a=[True, False], size=(np.sum(~mask.data.numpy())), 
                                     p=[ratio, 1.0-ratio]))
                
                #set weights to zero according to mask
                edge_weight = net.state_dict()[n]
                edge_weight[mask] = 0

            if prune_bias and n.endswith('bias'):
                
                #update mask to set a number of remaining False values to True according to ratio
                mask = locked_masks[n]
                mask[~mask] = torch.from_numpy(
                    np.random.choice(a=[True, False], size=(np.sum(~mask.data.numpy())), 
                                     p=[ratio, 1.0-ratio]))
                
                #set bias to zero according to mask
                unit_weight = net.state_dict()[n]
                unit_weight[mask] = 0
    
    else:

        # init functions if not defined
        if prune_weight and function is None:
            function = lambda ew, uw: torch.abs(ew)

        if prune_bias and function_bias is None:
            function_bias = lambda ew, uw: torch.sum(torch.abs(ew),dim=1) + torch.abs(uw)

        if prune_across_layers:
            if prune_weight and threshold is None:
                
                #get function output and prune mask in 1-dim format
                fouts, mask_outs = get_fouts(net,'weight',function,locked_masks)
                fall = torch.cat([torch.flatten(fouts[i],start_dim=0,end_dim=-1) for i in range(len(fouts))])
                mask_all = torch.cat([torch.flatten(mask_outs[i],start_dim=0,end_dim=-1) for i in range(len(mask_outs))])
                
                # compute number of edges to be pruned
                prune_num = int(torch.round(torch.sum(~mask_all*ratio))) ###TODO need mask in 
                
                # compute threshold
                fallc = fall.clone()[~mask_all]
                size1 = np.product(list(fallc.size()))
                threshold, _ = torch.sort(fallc.view(size1, -1),dim=0)
                threshold = float(threshold[prune_num])
                #print(prune_num,threshold)
                
            if prune_bias and threshold_bias is None:
                
                #get function output and prune mask in 1-dim format
                fouts_bias, mask_outs_bias = get_fouts(net,'bias',function_bias,locked_masks)
                fall = torch.cat([torch.flatten(fouts_bias[i],start_dim=0,end_dim=-1) for i in range(len(fouts))])
                mask_all = torch.cat([torch.flatten(mask_outs_bias[i],start_dim=0,end_dim=-1) for i in range(len(mask_outs_bias))])
                
                # compute number of edges to be pruned
                prune_num = int(torch.round(torch.sum(~mask_all*ratio)))
                
                # compute threshold
                fallc = fall.clone()[~mask_all]
                size1 = np.product(list(fallc.size()))
                threshold_bias, _ = torch.sort(fallc.view(size1, -1),dim=0)
                threshold_bias = float(threshold_bias[prune_num])
                
        if prune_weight:
            if threshold is None:
            
                # compute number of edges to be pruned
                for n, w in net.named_parameters():
                    if n.endswith('weight'):

                        mask = locked_masks[n]
                        edge_weight = net.state_dict()[n]
                        #edge_weight_c = edge_weight.clone()

                        # compute number of edges to be pruned
                        prune_num = int(torch.round(torch.sum(~mask*ratio)))

                        #get function output
                        fout = get_fout(n,net,function)

                        # compute corresponding threshold
                        foutc = fout.clone()[~mask]
                        size1 = np.product(list(foutc.size()))
                        threshold, _ = torch.sort(foutc.view(size1, -1),dim=0)
                        threshold = float(threshold[prune_num])

                        #update mask and set weights to zero
                        mask[fout < threshold] = True  
                        edge_weight[mask] = 0
            else:
                
                #apply threshold
                for n, w in net.named_parameters():
                    if prune_weight and n.endswith('weight'):                         

                        fout = get_fout(n,net,function)
                        mask = locked_masks[n]
                        edge_weight = net.state_dict()[n]
                        
                        mask[fout < threshold] = True 
                        edge_weight[mask] = 0
                                
        if prune_bias:
            if threshold_bias is None:
            
                # compute number of nodes to be pruned
                for n, w in net.named_parameters():
                    if n.endswith('bias'):

                        mask = locked_masks[n]
                        unit_weight = net.state_dict()[n]
                        #unit_weight_c = unit_weight.clone()

                        # compute number of nodes to be pruned
                        prune_num = int(torch.round(torch.sum(~mask*ratio)))

                        #get function output
                        fout = get_fout(n,net,function_bias)

                        # compute corresponding threshold
                        foutc=fout.clone()[~mask]
                        size1 = np.product(list(foutc.size()))
                        threshold_bias, _ = torch.sort(foutc.view(size1, -1),dim=0)
                        threshold_bias = float(threshold_bias[prune_num])  

                        #update mask and set weights to zero
                        mask[fout < threshold_bias] = True
                        unit_weight[mask] = 0
            else:
                
                #apply threshold
                for n, w in net.named_parameters():

                    if prune_bias and n.endswith('bias'):

                        fout = get_fout(n,net,function_bias)
                        mask = locked_masks[n]
                        unit_weight = net.state_dict()[n]
                        
                        mask[fout < threshold_bias] = True  
                        unit_weight[mask] = 0
