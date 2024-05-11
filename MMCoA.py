# import
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/Ad_cifar100_PGD.yaml')
parser.add_argument('--gpu',default=None)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--lamda',type=float,default=0.5,help='for adv img')
parser.add_argument('--lamda_advtxt',type=float,default=0.5,help='for adv txt')
parser.add_argument('--std2txt',action="store_true")
parser.add_argument('--print2txt',action="store_false")
parser.add_argument('--attack_domain',type=str,default="both1")
parser.add_argument("--dataset",type=str,default="cifar100",choices=["imagenet","cifar10","cifar100","tiny-imagenet"])
parser.add_argument('--start_epoch',type=int,default=0)
parser.add_argument("--resume_test",action="store_true")
parser.add_argument("--ckpt_dir",type=str,default=None,help="for retraining")


args = parser.parse_args()
gpu=args.gpu
seed=args.seed

import clip
from models.tokenization_bert import BertTokenizer
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torchvision import transforms
import ruamel.yaml as yaml
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
from attack import *
from torch.utils.data import Subset
from utils_2 import load_imagenet_folder2name,refine_classname
from tqdm import tqdm
import torch.nn as nn
import sys
import time
from torchvision.datasets import CIFAR100,CIFAR10,ImageFolder
from torch.utils.data import Dataset
from models.prompters import TokenPrompter, NullPrompter
from models.model import *
from attacks import attack_pgd

class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_per_class):
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class
        self.class_indices = [[] for _ in range(len(dataset.classes))]
        for i, label in enumerate(dataset.targets):
            self.class_indices[label].append(i)

    def __iter__(self):
        for i in range(len(self.class_indices)):
            random.shuffle(self.class_indices[i])
        for i in range(len(self.class_indices)):
            for _ in range(self.num_samples_per_class):
                if self.class_indices[i]:
                    yield self.class_indices[i].pop()
    def __len__(self):
        return self.num_samples_per_class*len(self.dataset.classes)


def calculate_accuracy(predicted_labels, true_labels):
    if len(predicted_labels) != len(true_labels):
        raise ValueError("The lengths of predicted labels and true labels do not match.")

    correct_predictions = 0
    total_predictions = len(predicted_labels)

    for pred, true in zip(predicted_labels, true_labels):
        if int(pred) == int(true):
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def test_imagadv(adv):
    """
    adv=1: only attack image
    adv=2: only attack text
    adv=3: attack image and text
    """
    num_image = len(test_loader.dataset)
    num_text = len(test_loader.dataset.text)
    real_labels = torch.zeros(num_image)
    image_feats = torch.zeros(num_image, model.module.visual.output_dim).to(device)
    text_feats = torch.zeros(num_text, model.module.visual.output_dim).to(device)
    if adv == 2 or adv==3:
         with torch.no_grad():
            texts = Mulmodal_attacker.text_attack(model, test_loader.dataset.text)
    else:
        texts = test_loader.dataset.text
    text_tokens = clip.tokenize(texts).cuda()
    labels_texts_tokens = clip.tokenize(test_loader.dataset.text).to(device)
    with torch.no_grad():
        text_features = model.module.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_feats = text_features.float().detach()
        
    for images, images_ids,label in test_loader:
        images = images.to(device)
        real_labels[images_ids] = label.to(torch.float32)
        label = label.to(device)
        
        if  adv == 1 or  adv == 3:
            images = Mulmodal_attacker.image_attack(model=model,images=images,targets=label,text_tokens=labels_texts_tokens)
        with torch.no_grad():
            images = clip_img_preprocessing(images)
            image_features = model.module.encode_image(images,None)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            

            image_feats[images_ids] = image_features.float().detach()
    
    sims_matrix = image_feats @ text_feats.t()
    text_probs = (100.0 * sims_matrix).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1) 
    acc = calculate_accuracy(top_labels.reshape(-1).tolist(),real_labels.tolist())
    return acc

class Imagetext_dataset(Dataset):
    
    def __init__(self, image_root,transform,train=False,datasize=None,random_choice=False):
        self.datasize = datasize
        if args.dataset == "cifar100":
            self.data = CIFAR100(image_root, transform=transform, download=True,train=train)
            class_names = self.data.classes
        elif args.dataset == "imagenet" or args.dataset == "tiny-imagenet":
            self.data = ImageFolder(image_root,transform=transform,)
            folder2name = load_imagenet_folder2name("imagenet_classes_names.txt")
            class_names = self.data.classes
            new_class_names = []
            for each in class_names:
                new_class_names.append(folder2name[each])
    
            class_names = new_class_names
            
        elif args.dataset == "cifar10":
            self.data = CIFAR10(image_root,transform=transform,download=True,train=train)
            class_names = self.data.classes
        indices = []
        self.class_names = refine_classname(class_names)

        if self.datasize is not None:

            if  not random_choice:
                self.data_subset = self.data
                self.sampler =  CustomSampler(self.data, num_samples_per_class=self.datasize)

            else:
                indices = torch.randperm(len(self.data))[:datasize]  
                self.data_subset = Subset(self.data, indices)
                self.sampler = None
            
                
        else:
            self.data_subset = self.data
            self.sampler = None

        self.text = [f"This is a photo of a {label}" for label in self.class_names]
        self.train = train
    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, index):
        image,label = self.data_subset[index] 
        return image,index,label

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

class Image_Text_attack():

    def __init__(self,imageattack_type="pgd",textattack_type="kl",device='cuda',**kwargs):

        self.imageattack_type = imageattack_type
        self.textattack_type = textattack_type
        self.norm_type = kwargs.get('norm_type')
        self.epsilon = kwargs.get('epsilon')/255.
        self.step_size = kwargs.get('step_size')/255.
        self.num_classes = kwargs.get('num_classes')
        self.num_iters = kwargs.get('num_iters')
        if imageattack_type == 'pgd':
            self.image_attacker = "attack_pgd"
            self.I_criterion = torch.nn.CrossEntropyLoss().to(device)

        if textattack_type=="kl":
            self.text_attacker = BertAttack(ref_model, tokenizer, cls=cls)


    def image_attack(self,model,images,targets,text_tokens=None):
        if self.imageattack_type == 'pgd':
            images_adv = globals()[self.image_attacker](prompter,model,add_prompter,
                                                        self.I_criterion,images,targets,text_tokens,
                                        self.step_size,self.num_iters,
                                    self.norm_type,epsilon=self.epsilon
                                   ) + images
        else:
            raise ValueError
        return images_adv


    def text_attack(self,model,texts,**kwargs):
        if self.textattack_type=='kl':
            texts_adv = self.text_attacker.attack(model, texts)
        else:
            raise ValueError

        return texts_adv
    






if __name__=="__main__":
    cls = False
    config = args.config
    yaml = yaml.YAML()
    output_dir=config['output_dir']
    image_encoder=config['image_encoder']
    text_encoder= config['text_encoder']
    adv_train = config['adv_train']
    for key,value in vars(args).items():
        config[key] = value
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(output_dir, 'config.yaml'), 'w'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    text_encoder= config['text_encoder']
    add_prompt_len = 0
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    model, preprocess = clip.load(image_encoder, device, jit=False, prompt_len=add_prompt_len)
    model_text, model_image = None, None
    model.set_tokenizer(tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder).to(device)

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)  # .to(device)
    model.eval()

    prompter = NullPrompter()  # .to(device)
    add_prompter = TokenPrompter(add_prompt_len)  # .to(device)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    if args.resume_test:
        print("resuming from checkpoint\n")
        checkpoint = torch.load(args.ckpt_dir)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch']+1

   

    print("Creating dataset")
    if hasattr(model, "module"):
        n_px = model.module.visual.input_resolution
    else:   
        n_px = model.visual.input_resolution

   
    preprocess = transforms.Compose([
    transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),  # because of random size in the original image
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    if args.dataset == "cifar100" or args.dataset == "cifar10" or args.dataset == "tiny-imagenet":
        train_transform = preprocess   
        test_transform = preprocess
    elif args.dataset == "imagenet":
        train_transform = preprocess224
        test_transform = preprocess224
    
    train_dataset = Imagetext_dataset(image_root=config['train_image_root'],transform=train_transform,datasize=config['train_size'],train=True,random_choice=config['train_random_choice'])
    if train_dataset.sampler is not None:
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], num_workers=64,sampler=train_dataset.sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], num_workers=64,shuffle=True)
    
    test_dataset = Imagetext_dataset(image_root=config['test_image_root'],transform=test_transform,datasize=config['test_size'],train=False,random_choice=config['test_random_choice'])
    if test_dataset.sampler is not None:
         test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=64,sampler=test_dataset.sampler)
    else:
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=64,shuffle=False)

    config['num_classes'] = len(test_dataset.class_names)
    Mulmodal_attacker = Image_Text_attack(device=device,**config)
    # model funtuning and trainning
    lr = config['optimizer']['lr']
    if not config['full_tuning']:
        print('-----------------\n')
        
        if config["attack_domain"] == "text":
           
            model.module.token_embedding.train()
            model.module.transformer.train()
            model.module.ln_final.train()
            

            optimizer = torch.optim.Adam([{'params':model.module.token_embedding.parameters()},
                            {'params':model.module.transformer.parameters()},{'params':model.module.ln_final.parameters()}], 
                                            lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    else:
        print('**********************\n')
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    labels_texts = train_dataset.text
    labels_texts_tokens = clip.tokenize(labels_texts).to(device)
    num_epochs = config['num_epoches']
    best_cleanacc = 0
    best_advacc = 0
    acc = 0
    adv_acc = 0
    START_EPOCH = args.start_epoch
    log_file = os.path.join(config['output_dir'],config['log_name'])
    if args.resume or args.resume_test:
        txt_mode = 'a'
    else:
        txt_mode = "w"
    start_time = time.time()
    with open(log_file, txt_mode) as file:
        default_stdout = sys.stdout
        default_stderr = sys.stderr
        if args.print2txt:
            sys.stdout = file
        if args.std2txt:
            sys.stderr = file
        for epoch in range(START_EPOCH,num_epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            for images, images_ids,labels in pbar:
                optimizer.zero_grad()
                clean_images= images.to(device)
                clean_texts = [f"This is a photo of a {test_dataset.class_names[label_item]}" for label_item in labels]
                labels = labels.to(device)
                if adv_train:
                    clean_text_tokens = clip.tokenize(clean_texts).to(device)
                    if config['attack_domain']=="both1" : 
                        images_advs = Mulmodal_attacker.image_attack(model,clean_images,labels,labels_texts_tokens)
                        logits_adimg2cltxt, logits_cltxt2adimg = multiGPU_CLIP(model,clip_img_preprocessing(images_advs), clean_text_tokens,None)
                    if config['attack_domain']=="text" or config['attack_domain']=="both1" :
                        texts_advs = Mulmodal_attacker.text_attack(model,clean_texts)
                        attack_text_tokens = clip.tokenize(texts_advs).to(device)
                        logits_climg2_adtxt, logits_adtxt2climg = multiGPU_CLIP(model,clip_img_preprocessing(clean_images), attack_text_tokens,None) 
                        
                    # Compute loss
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                   
                    if config['attack_domain']=="text":
                        total_loss = 0.5*(loss_img(logits_climg2_adtxt,ground_truth) + loss_txt(logits_adtxt2climg,ground_truth))
                        print('advtxt')
                    elif config['attack_domain']=="both1" :
                        total_loss = (args.lamda_advtxt*(loss_img(logits_climg2_adtxt,ground_truth) + loss_txt(logits_adtxt2climg,ground_truth)) +
                                    args.lamda*(loss_img(logits_adimg2cltxt,ground_truth) + loss_txt(logits_cltxt2adimg,ground_truth)))


    # clean image & clean text
                else:
                    # Forward pass                    
                    text_tokens = clip.tokenize(clean_texts).to(device)
                    clean_images = clip_img_preprocessing(clean_images)

                    logits_per_image, _ = model(clean_images, labels_texts_tokens)
                    total_loss = loss_img(logits_per_image,labels)
                total_loss.backward()             
                optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")

            acc = test_imagadv(adv=0)
            print('the clean acc for testdataset is,',acc)
            
        end_time = time.time()
        print(f"Time used:{end_time-start_time}\n")
    sys.stdout = default_stdout
    sys.stderr = default_stderr
    print('over')


