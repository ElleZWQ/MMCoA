import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/Ad_cifar100_PGD.yaml')
parser.add_argument('--gpu',default=None)
parser.add_argument('--seed',default=42)
parser.add_argument('--std2txt',action="store_true")
parser.add_argument('--attack_domain',type=str,default="image",choices=["image","text","both1","all","both2","both3"])
parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
parser.add_argument("--dataset",type=str,default="cifar100")
parser.add_argument("--cls",type=bool,default=False)
parser.add_argument("--mix_alpha", type=float, default=-1,
                        help="interpolation")
parser.add_argument("--resume_test",action="store_true")
parser.add_argument("--ckpt_dir",type=str,default=None,help="for retraining")
parser.add_argument('--log_resume',action='store_true')
args = parser.parse_args()
gpu=args.gpu
seed=args.seed
from utils import convert_models_to_fp32, refine_classname,load_imagenet_folder2name
import torch
import clip
import numpy as np
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Subset
import random
from pathlib import Path
import sys
from models.prompters import TokenPrompter, NullPrompter
from attack import BertAttack,MultiModalAttacker,ImageAttacker
from transformers import BertForMaskedLM
from models.tokenization_bert import BertTokenizer
import torch.backends.cudnn as cudnn
from torchvision.datasets import *
from torch.utils.data import Dataset
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(MEAN).view(3, 1, 1).cuda()
std = torch.tensor(STD).view(3, 1, 1).cuda()
upper_limit, lower_limit = 1, 0

def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text
def normalize(X):
    return (X - mu) / std


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
def attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

class Imagetext_dataset(Dataset):
    
    def __init__(self, image_root,transform,train=False,datasize=None,random_choice=False,):
        self.datasize = datasize
        self.subset_sampler =  None
        if args.dataset == "cifar100":
            self.data = CIFAR100(image_root, transform=transform, download=True,train=train)
            class_names = self.data.classes
        elif args.dataset == "Food101":
            if train:
                split='train'
            else:
                split='test'
            self.data = Food101(image_root, transform=transform, download=True,split=split)
            class_names = self.data.classes
        elif args.dataset == "Caltech101":
            self.data = Caltech101(image_root, target_type='category', transform=transform,
                                             download=True)
            class_names = self.data.categories.copy()
        elif args.dataset == "EuroSAT":
            self.data = EuroSAT(image_root,transform=transform, download=True)
        elif args.dataset == "STL10":
            self.data = STL10(image_root,split='test',transform=transform, download=True)
            class_names = self.data.classes.copy()
        elif args.dataset == "SUN397":
            self.data = SUN397(image_root,transform=transform, download=True)
            class_names = self.data.classes.copy()
        elif args.dataset == "StanfordCars":
            self.data = StanfordCars(image_root,split='test',transform=transform, download=True)
            class_names = self.data.classes.copy()
        elif args.dataset == "oxfordpet":
            self.data = OxfordIIITPet(image_root,split='test',transform=transform, download=True)
        elif args.dataset == "Caltech256":
            self.data = Caltech256(image_root, transform=transform,
                                             download=True)
        elif args.dataset == "flowers102":
            self.data = Flowers102(image_root,split='test',transform=transform, download=True)
        elif args.dataset == "Country211":
            self.data = Country211(image_root,split='test',transform=transform, download=True)
        elif args.dataset == "dtd":
            self.data = DTD(image_root,split='test',transform=transform, download=True)
        elif args.dataset == "fgvc_aircraft":
            self.data = FGVCAircraft(image_root,split='test',transform=transform, download=True)
        elif args.dataset == "imagenet" or args.dataset=="tiny-imagenet":
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
        if hasattr(self.data, 'clip_prompts'):
            self.text = self.data.clip_prompts
            self.class_names = []
            for i,prompt in enumerate(self.text):
                words = prompt.split()
                last_word = words[-1]
                self.class_names.append(last_word)
        else:
            self.class_names = refine_classname(class_names)
            self.text = [f"This is a photo of a {label}" for label in self.class_names]

        if self.datasize is not None:
            if  not random_choice:
                if args.dataset == "imagenet" or args.dataset == "tiny-imagenet":
                    num_classes = len(self.data.classes)
                    class_indices = [[] for _ in range(num_classes)]
                    for idx, (image, label) in enumerate(self.data.imgs):
                        class_indices[label].append(idx)
                    subset_indices = []
                    for indices in class_indices:
                        subset_indices.extend(torch.randperm(len(indices))[:datasize])
                    self.subset_sampler = SubsetRandomSampler(subset_indices)
                    self.data_subset = self.data
                                        
                else:
                    for class_index,_ in enumerate(self.class_names):
                        try:
                            class_indices = torch.where(torch.tensor(self.data.targets)==class_index)[0]
                        except:
                            pass 
                        class_indices = class_indices.tolist()
                        selected_indices = random.sample(class_indices, self.datasize)
                        indices.append(selected_indices)
                    self.data_subset = Subset(self.data, indices)


            else:
                indices = torch.randperm(len(self.data))[:datasize]  
                self.data_subset = Subset(self.data, indices)
            
                
        else:
            self.data_subset = self.data
        
        
        self.train = train
    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, index):
        image,label = self.data_subset[index] 
        if image.size(0) != 3:
            image = image.expand(3, -1, -1)

        return image,index,label

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


# class attack
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
        elif imageattack_type == "kl":
            self.image_attacker = ImageAttacker(self.epsilon,norm_type=self.norm_type,
                                           preprocess=clip_img_preprocessing, bounding=(0, 1),
                                           cls=cls,step_size=self.step_size)

        if textattack_type=="kl":
            self.text_attacker = BertAttack(ref_model, tokenizer, cls=cls)


    def image_attack(self,model,images,targets,text_tokens=None):
        if self.imageattack_type == 'pgd':
            images_adv = globals()[self.image_attacker](prompter,model,model_text,model_image,add_prompter,
                                                        self.I_criterion,images,targets,text_tokens,
                                        self.step_size,self.num_iters,
                                    self.norm_type,epsilon=self.epsilon
                                   ) + images

        elif self.imageattack_type == 'kl':
            images_adv = self.image_attacker.run_trades(model, images, self.num_iters,None)
        else:
            raise ValueError
        return images_adv


    def text_attack(self,model,texts,**kwargs):
        if self.textattack_type=='kl':
            texts_adv = self.text_attacker.attack(model, texts)
        else:
            raise ValueError

        return texts_adv
    def multimodal_attack(self,model,images,texts):
        if self.textattack_type=='kl' and self.imageattack_type == 'kl':
            multi_attacker = MultiModalAttacker(model, self.image_attacker, self.text_attacker, tokenizer, cls=args.cls)
            images_adv, texts_adv = multi_attacker.run_before_fusion(images, texts, adv=4, num_iters=config['num_iters'], max_length=77,
                                                             alpha=3)
            return images_adv,texts_adv

        else:
            raise ValueError




def test_imagadv(adv):
    num_image = len(test_loader.dataset)
    num_text = len(test_loader.dataset.text)
    real_labels = torch.zeros(num_image)
    image_feats = torch.zeros(num_image, model.module.visual.output_dim).to(device)
    text_feats = torch.zeros(num_text, model.module.visual.output_dim).to(device)
    if adv == 2 or adv==3 or adv ==4:
        with torch.no_grad():
            texts = Mulmodal_attacker.text_attack(model, test_loader.dataset.text)
            # print(texts)
    else:
        texts = test_loader.dataset.text
    text_tokens = clip.tokenize(texts).to(device)
    labels_texts_tokens = clip.tokenize(test_loader.dataset.text).to(device)
    with torch.no_grad():
        text_features = model.module.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_feats = text_features.float().detach()
    
    for images, images_ids,label in test_loader:
        images = images.to(device)
        real_labels[images_ids] = label.to(torch.float32)
        label = label.to(device)
        clean_texts = [f"This is a photo of a {test_dataset.class_names[label_item]}" for label_item in label]
        if  adv == 1 or  adv == 3:
            images = Mulmodal_attacker.image_attack(model=model,images=images,targets=label,text_tokens=labels_texts_tokens)
        if adv == 4:
            images,adv_texts = Mulmodal_attacker.multimodal_attack(model=model,images=images,texts=clean_texts)
            
        with torch.no_grad():
            images = clip_img_preprocessing(images) 
            image_features = model.module.encode_image(images,ind_prompt = None)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            

            image_feats[images_ids] = image_features.float().detach()
    
    sims_matrix = image_feats @ text_feats.t()
    text_probs = (100.0 * sims_matrix).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1) 
    acc = calculate_accuracy(top_labels.reshape(-1).tolist(),real_labels.tolist())
    return acc




if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    output_dir=config['output_dir']
    for key,value in vars(args).items():
        config[key] = value
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(output_dir, 'config.yaml'), 'w'))

    text_encoder= config['text_encoder']
    cls = args.cls
    add_prompt_len = 0
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
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
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            if args.mix_alpha > 0:
                alpha = args.mix_alpha
                checkpoint_ori = torch.load('original_clip.pth.tar')
                theta_ori = checkpoint_ori['vision_encoder_state_dict']
                theta_rob = checkpoint['vision_encoder_state_dict']

                theta = {
                    key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key]
                    for key in theta_ori.keys()
                }
                model.module.visual.load_state_dict(theta)

            else:

                model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.resume_test:
        print("resuming from checkpoint\n")
        checkpoint = torch.load(args.ckpt_dir)
        model.load_state_dict(checkpoint['model'])
    
   

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
    
    if args.dataset == "cifar100" or args.dataset == "cifar10" or args.dataset == "STL10":
        test_transform = preprocess
    elif args.dataset == "imagenet" or args.dataset == "Food101" or args.dataset== "Caltech101" or args.dataset == "EuroSAT" or args.dataset == "tiny-imagenet" or args.dataset == "PCAM" \
    or args.dataset == "SUN397" or args.dataset == "StanfordCars" or args.dataset == "oxfordpet" or args.dataset == "Caltech256" \
    or args.dataset == "flowers102" or args.dataset == "Country211" or args.dataset == "dtd" or args.dataset == "fgvc_aircraft"   :
        test_transform = preprocess224

   

    test_dataset = Imagetext_dataset(image_root=config['image_root'],transform=test_transform,datasize=config['test_size'],train=False,random_choice=config['random_choice'])
    if test_dataset.subset_sampler is not None:
         test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=8,sampler=test_dataset.subset_sampler)
    else:
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=8,shuffle=False)

    labels_texts = test_loader.dataset.text
    labels_texts_tokens = clip.tokenize(labels_texts).to(device)
    config['num_classes'] = len(test_dataset.text)
    if args.attack_domain == "both2":
        imageattack_type="kl"
        textattack_type="kl"
    else:
        imageattack_type="pgd"
        textattack_type="kl"
    Mulmodal_attacker = Image_Text_attack(imageattack_type,textattack_type,device=device,**config)
    log_file = os.path.join(config['output_dir'],config['log_name'])
    if args.resume_test or args.attack_domain=="both2" or args.resume or args.log_resume:
            txt_mode = 'a'
    else:
        txt_mode = "w"
    with open(log_file, txt_mode) as file:
        default_stdout = sys.stdout
        default_stderr = sys.stderr      
        sys.stdout = file
        if args.std2txt:
            sys.stderr = file
        print(f"The dataset: {args.dataset}\n")
        if args.attack_domain == "all":
            if args.ckpt_dir is not None:
                 print(f"It is using the base ckpt from {args.ckpt_dir}\n")
            if args.resume is not None:
                print(f"It is using the base ckpt from {args.resume}\n")
            print(f"The eps: {config['epsilon']}, the num_iters: {config['num_iters']},the step size: {config['step_size']}\n")

            for adv in range(4):
                acc = test_imagadv(adv)
                print(f"It is test acc for adv:{adv}, dataset:{args.dataset} and the acc is {acc}\n")
        elif args.attack_domain == 'both3':
            if args.ckpt_dir is not None:
                 print(f"It is using the base ckpt from {args.ckpt_dir}\n")
            if args.resume is not None:
                print(f"It is using the base ckpt from {args.resume}\n")
            print(f"The eps: {config['epsilon']}, the num_iters: {config['num_iters']},the step size: {config['step_size']}\n")
            adv_list = [1,3]
            for adv in adv_list:
                acc = test_imagadv(adv)
                print(f"It is test acc for adv:{adv}, dataset:{args.dataset} and the acc is {acc}\n")
        else:
            if args.attack_domain=="image":
                adv = 1
            elif args.attack_domain=="text":
                adv = 2
            elif args.attack_domain=="both1":
                adv = 3
            elif args.attack_domain=="both2":
                adv = 4
            if not args.resume_test:
                print(f"It is using the ckpt from {args.resume}\n")
            else:
                print(f"It is using the base ckpt from {args.ckpt_dir}\n")
            print(f"The eps: {config['epsilon']}, the num_iters: {config['num_iters']},the step size: {config['step_size']}\n")
            print(f"It is test acc for adv:{adv}, dataset:{args.dataset}\n")

            adv_acc = test_imagadv(adv)
            print(f"It is testing the adv acc {adv_acc} for {args.attack_domain}\n")
            acc = test_imagadv(adv=0)
            print(f"It is testing the clean acc {acc}\n")
        print('==================================\n')
        

