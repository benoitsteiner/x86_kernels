# Importing Libraries
import time
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
import math
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
from fairseq import (
    checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
)
import fairseq_local.metrics as fmetrics
import fairseq.utils as fairsequtils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter
import ncf.data_utils
import ncf.model
import ncf.evaluate
import ncf.config

# Custom Libraries
import utils

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    num_workers=0

    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, mobilenet, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet, mobilenet 
    
    elif args.dataset == "imagenet":
        traindataset = datasets.ImageNet('../data', train=True, download=True,transform=transform)
        testdataset = datasets.ImageNet('../data', train=False, transform=transform)
        from archs.imagenet import AlexNet, fc1, LeNet5, vgg, resnet
 
    elif args.dataset == "ncf":
        train_data, test_data, user_num ,item_num, train_mat = ncf.data_utils.load_all()
        traindataset = ncf.data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
        testdataset = ncf.data_utils.NCFData(test_data, item_num, train_mat, 0, False)
        num_workers=4

    elif args.dataset == "iwslt14":
        fairsequtils.import_user_module(args)
        task = tasks.setup_task(args)
        valid_subset='valid'
        task.load_dataset(valid_subset, combine=False, epoch=0)
        #model = task.build_model(args)
        #criterion = task.build_criterion(args)
        #trainer = Trainer(args, task, model, criterion)
        traindataset = None
        testdataset = None

    else:
        print("\nWrong Dataset choice \n")
        exit()

    if traindataset:
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,drop_last=False)
        #train_loader = cycle(train_loader)
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet50":
        model = resnet.ResNet50().to(device)   
    elif args.arch_type == "mobilenet":
        model = mobilenet.mobilenet_v2().to(device)
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    elif args.arch_type == "transformer":
        model = task.build_model(args)
   # If you want to add extra model paste here
    elif args.arch_type == "neumf":
        model = ncf.model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                              args.dropout, ncf.config.model, None, None)

    # Weight Initialization
    model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    if args.arch_type == "neumf":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
    elif args.arch_type != "transformer":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
    else:
        criterion = task.build_criterion(args)
        trainer = Trainer(args, task, model, criterion)
        assert(trainer.model)
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)


    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)  
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)   
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)   
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                if args.arch_type == "neumf":
                    HR, NDCG = ncf.evaluate.metrics(model, test_loader, args.top_k)
                    accuracy = HR
                elif args.arch_type != "transformer":
                    accuracy = test(model, test_loader, criterion)
                else:
                    # Hack
                    accuracy = best_accuracy + 1
                    #accuracy = test_nmt(args, trainer, task, epoch_itr)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            if args.arch_type == "neumf":
                loss = train_neumf(model, train_loader, test_loader, optimizer, criterion)
            elif args.arch_type != "transformer":
                loss = train(model, train_loader, optimizer, criterion)
            else:
                loss = train_nmt(args, trainer, task, epoch_itr)

            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    
   
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

def train_neumf(model, train_loader, test_loader, optimizer, criterion):
    for epoch in range(args.epochs):
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
                label = label.float()  #.cuda()

                model.zero_grad()
                prediction = model(user, item)
                loss = criterion(prediction, label)
                loss.backward()
                optimizer.step()
                # writer.add_scalar('data/loss', loss.item(), count)
                #count += 1

        model.eval()
        HR, NDCG = ncf.evaluate.metrics(model, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                        time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
    return loss


# For NMT only
def train_nmt(args, trainer, task, epoch_itr):
   # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    valid_subsets = ['valid']
    max_update = args.max_update or math.inf
    for samples in progress:
        with fmetrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            num_updates = trainer.get_num_updates()
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = get_training_stats('train_inner')
            progress.log(stats, tag='train', step=num_updates)

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    return 0

def test_nmt(args, trainer, task, epoch_itr):
    return 0

def get_training_stats(stats_key):
    stats = fmetrics.get_smoothed_values(stats_key)
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    #stats['wall'] = round(fmetrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=="__main__":
    
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrate",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")

    # For NMT
    parser.add_argument("--arch", default='transformer_iwslt_de_en', type=str)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--task", default='translation', type=str)
    parser.add_argument("--left_pad_source", default='True')
    parser.add_argument("--left_pad_target", default='False')
    parser.add_argument("--data", default='data-bin/iwslt14.tokenized.de-en', type=str)
    parser.add_argument("--source_lang", default=None)
    parser.add_argument("--target_lang", default=None)
    parser.add_argument("--dataset_impl", default=None)
    parser.add_argument("--upsample_primary", default=1)
    parser.add_argument("--max_source_positions", default=1024)
    parser.add_argument("--max_target_positions", default=1024)
    parser.add_argument("--load_alignments", default=False)
    parser.add_argument("--truncate_source", default=False)
    parser.add_argument("--fast_stat_sync", default=False)
    parser.add_argument("--distributed_rank", default=0)
    parser.add_argument("--save_dir", default="checkpoints/transformer")
    parser.add_argument("--restore_file", default='checkpoint_last.pt')
    parser.add_argument("--reset_optimizer", default=False)     # FIXME
    parser.add_argument("--reset_meters", default=False)     # FIXME
    parser.add_argument("--reset_lr_scheduler", default=False)  # FIXME
    parser.add_argument("--optimizer_overrides", default='{}')
    parser.add_argument("--train_subset", default='train')
    parser.add_argument("--max_tokens", default=4000)
    parser.add_argument("--max_sentences", default=None)
    parser.add_argument("--distributed_world_size", default=1)
    parser.add_argument("--required_batch_size_multiple", default=8)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--num_workers", default=1)
    parser.add_argument("--encoder_layers_to_keep", default=None)
    parser.add_argument("--decoder_layers_to_keep", default=None)
    parser.add_argument("--encoder_layerdrop", default=0)
    parser.add_argument("--decoder_layerdrop", default=0)
    parser.add_argument("--criterion", default='cross_entropy')
    parser.add_argument("--use_bmuf", default=False)
    parser.add_argument("--optimizer", default='nag')
    parser.add_argument("--lr", default=[0.25])
    parser.add_argument("--lr_scheduler", default='fixed')
    parser.add_argument("--lr_shrink", default=0.1)
    parser.add_argument("--force_anneal", default=None)
    parser.add_argument("--fix_batches_to_gpus", default=False)
    parser.add_argument("--curriculum", default=0)
    parser.add_argument("--update_freq", default=[1])
    parser.add_argument("--log_format", default=None)
    parser.add_argument("--no_progress_bar", default=False)
    parser.add_argument("--log_interval", default=1000)
    parser.add_argument("--tensorboard_logdir", default='')
    parser.add_argument("--max_update", default=0)
    parser.add_argument("--sentence_avg", default=False)
    parser.add_argument("--clip_norm", default=0.1)
    parser.add_argument("--empty_cache_freq", default=0)
    parser.add_argument("--disable_validation", default=False)
    parser.add_argument("--save_interval_updates", default=0)
    parser.add_argument("--cpu", default=False)

    #Namespace(activation_dropout=0.0, activation_fn='relu', adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, arch='transformer_iwslt_de_en', attention_dropout=0.0, best_checkpoint_metric='loss', bpe=None, bucket_cap_mb=25, clip_norm=0.1, cpu=False, criterion='cross_entropy', cross_self_attention=False, curriculum=0, data='data-bin/iwslt14.tokenized.de-en', dataset_impl=None, ddp_backend='c10d', decoder_attention_heads=4, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=1024, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.2, empty_cache_freq=0, encoder_attention_heads=4, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=1024, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=False, encoder_normalize_before=False, fast_stat_sync=False, find_unused_parameters=False, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, keep_interval_updates=-1, keep_last_epochs=-1, layer_wise_attention=False, layernorm_embedding=False, lazy_load=False, left_pad_source='True', left_pad_target='False', load_alignments=False, log_format=None, log_interval=1000, lr=[0.25], lr_scheduler='fixed', lr_shrink=0.1, max_epoch=0, max_sentences=None, max_sentences_valid=None, max_source_positions=1024, max_target_positions=1024, max_tokens=4000, max_tokens_valid=4000, max_update=0, maximize_best_checkpoint_metric=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1, momentum=0.99, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_token_positional_embeddings=False, num_workers=1, optimizer='nag', optimizer_overrides='{}', raw_text=False, required_batch_size_multiple=8, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints/fconv', save_interval=1, save_interval_updates=0, seed=1, sentence_avg=False, share_all_embeddings=False, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, source_lang=None, target_lang=None, task='translation', tensorboard_logdir='', threshold_loss_scale=None, tokenizer=None, train_subset='train', truncate_source=False, update_freq=[1], upsample_primary=1, use_bmuf=False, user_dir=None, valid_subset='valid', validate_interval=1, warmup_updates=0, weight_decay=0.0)

    # For NCF
    #parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    #parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epochs",
        type=int,
        default=20,
        help="training epoches")
    parser.add_argument("--top_k",
        type=int,
        default=10,
        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
        type=int,
        default=32,
        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers",
        type=int,
        default=3,
        help="number of layers in MLP model")
    parser.add_argument("--num_ng",
        type=int,
        default=4,
        help="sample negative items for training")
    parser.add_argument("--test_num_ng",
        type=int,
        default=99,
        help="sample part of negative items for testing")
    parser.add_argument("--out",
        default=True,
        help="save model or not")
    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
