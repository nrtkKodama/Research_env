import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import math

import io
from PIL import Image
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import logging
from sklearn.model_selection import GroupKFold
log = logging.getLogger(__name__)

def my_LoadData(args):
    logging.info("Import SpermData ... ")
    SpermDataDict = pickle.load(open(args.sperm_data_path, "rb"))
    logging.info("-> Completed!")
    img_size = 224
    # if args.frame_select == 'all':
    #     dataset = SpermDatasetFrames(SpermDataDict, img_size=(img_size, img_size))
    # else:
    #     #TODO implement id split train/test
    #     dataset = SpermDatasetVideo(SpermDataDict, img_size=(img_size, img_size))
    dataset = SpermDatasetVideo(SpermDataDict, img_size=(img_size, img_size))
    return dataset


# データセット作成


class SpermDatasetVideo(torch.utils.data.Dataset):
    # structure of SpermDataset 
    def __init__(self, data_dict, img_size=(150, 150),num_frame=16):
        self.data_dict = data_dict
        n_sample = len(self.data_dict)
        self.videos = [0 for i in range(n_sample)]
        self.pos = [0 for i in range(n_sample)]
        self.grades = [0 for i in range(n_sample)]
        self.trajs = [0 for i in range(n_sample)]
        for i, key in enumerate(self.data_dict):
            self.videos[i] = self.data_dict[key]["Movie"]  # 16フレーム目を見ている
            self.pos[i] = self.data_dict[key]["Pos"]
            self.grades[i] = self.data_dict[key]["Grade"]
            self.trajs[i] = pos_to_td(self.pos[i])
        self.IDs = list(self.data_dict.keys())
        self.img_size = img_size
        self.num_frame = num_frame

    # len(SpermDataset) is len(IDs)
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, i):
        # mean = [0.45, 0.45, 0.45]
        # std = [0.225, 0.225, 0.225]
        # transform = Compose([
        #         UniformTemporalSubsample(self.num_frame),
        #         Lambda(lambda x: x/255.0),
        #         NormalizeVideo(mean, std),
        #         ShortSideScale(size=self.img_size)])
            
        # video = torch.from_numpy(self.videos[i].astype(np.float32)).permute(3,0,1,2)
        # video = transform(video)
        video=[]
        for i_video in self.videos[i]:
            if type(i_video) != Image.Image:
                image = Image.fromarray(i_video)
            else:
                image = i_video

            # PIL Image -> torch.Tensor(dtype=float32) & Resize by img_size
            img_size = self.img_size
            transform = transforms.Compose(
                [transforms.Resize(img_size),transforms.RandomRotation(degrees=[-180, 180]), transforms.ToTensor()]
            )
            image = transform(image).to(torch.float32)
            video.append(image)

        video=torch.stack(video)
        # array like -> torch.Tensor(dtype=float32)
        grade = torch.Tensor(self.grades[i])

        # array like -> torch.Tensor(dtype=float32)
        traj = torch.Tensor(np.array(self.trajs[i])).view(-1)  # 1次元Tensorに変換

        # str (No Use for Prediction)
        ID = str(self.IDs[i])

        return (
            video,
            traj,
            grade,
            ID,
        )  # {'ID' : ID, 'image' : image, 'traj' : traj, 'grade' : grade}

class SpermDatasetFrames(torch.utils.data.Dataset):
    # structure of SpermDataset 
    def __init__(self, data_dict, img_size=(150, 150),num_frame=16):
        self.data_dict = data_dict
        n_sample = len(self.data_dict)*num_frame
        self.images = [0 for i in range(n_sample)]
        self.pos = [0 for i in range(n_sample)]
        self.grades = [0 for i in range(n_sample)]
        self.trajs = [0 for i in range(n_sample)]
        i=0
        for _, key in enumerate(self.data_dict):
            for f in range(16):
                self.images[i] = self.data_dict[key]["Movie"][f]  # 16フレーム目を見ている
                self.pos[i] = self.data_dict[key]["Pos"]
                self.grades[i] = self.data_dict[key]["Grade"]
                self.trajs[i] = pos_to_td(self.pos[i])
                i+=1
        self.IDs = list(self.data_dict.keys())
        self.img_size = img_size
        self.num_frame = num_frame

    # len(SpermDataset) is len(IDs)
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, i):
         # array like -> PIL Image
        if type(self.images[i]) != Image.Image:
            image = Image.fromarray(self.images[i])
        else:
            image = self.images[i]

        # PIL Image -> torch.Tensor(dtype=float32) & Resize by img_size
        img_size = self.img_size
        transform = transforms.Compose(
            [transforms.Resize(img_size),transforms.RandomRotation(degrees=[-180, 180]), transforms.ToTensor()]
        )
        image = transform(image).to(torch.float32)
        image=torch.tensor(image)
        # array like -> torch.Tensor(dtype=float32)
        grade = torch.Tensor(self.grades[i])

        # array like -> torch.Tensor(dtype=float32)
        traj = torch.Tensor(np.array(self.trajs[i])).view(-1)  # 1次元Tensorに変換

        # str (No Use for Prediction)
        ID = str(self.IDs[i])

        return (
            image,
            traj,
            grade,
            ID,
        )  
        # return {'image' : image, 'traj' : traj, 'grade' : grade, 'ID' : ID}

class SpermDataset(torch.utils.data.Dataset):
    # structure of SpermDataset
    def __init__(self, data_dict, img_size=(150, 150)):
        self.data_dict = data_dict
        n_sample = len(self.data_dict)
        self.images = [0 for i in range(n_sample)]
        self.pos = [0 for i in range(n_sample)]
        self.grades = [0 for i in range(n_sample)]
        self.trajs = [0 for i in range(n_sample)]
        for i, key in enumerate(self.data_dict):
            self.images[i] = self.data_dict[key]["Movie"][0]  # 1フレーム目を見ている
            self.pos[i] = self.data_dict[key]["Pos"]
            self.grades[i] = self.data_dict[key]["Grade"]
            self.trajs[i] = pos_to_td(self.pos[i])
        self.IDs = list(self.data_dict.keys())
        self.img_size = img_size

    # len(SpermDataset) is len(IDs)
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, i):
        # array like -> PIL Image
        if type(self.images[i]) != Image.Image:
            image = Image.fromarray(self.images[i])
        else:
            image = self.images[i]

        # PIL Image -> torch.Tensor(dtype=float32) & Resize by img_size
        img_size = self.img_size
        transform = transforms.Compose(
            [transforms.Resize(img_size),transforms.RandomRotation(degrees=[-180, 180]), transforms.ToTensor()]
        )
        image = transform(image).to(torch.float32)

        # array like -> torch.Tensor(dtype=float32)
        grade = torch.Tensor(self.grades[i])

        # array like -> torch.Tensor(dtype=float32)
        traj = torch.Tensor(np.array(self.trajs[i])).view(-1)  # 1次元Tensorに変換

        # str (No Use for Prediction)
        ID = str(self.IDs[i])

        return (
            image,
            traj,
            grade,
            ID,
        )  # {'ID' : ID, 'image' : image, 'traj' : traj, 'grade' : grade}

class PosDataset(torch.utils.data.Dataset):#position as input 
    # structure of SpermDataset
    def __init__(self, data_dict, img_size=(224, 224)):
        self.data_dict = data_dict
        n_sample = len(self.data_dict)
        self.pos_images = [0 for i in range(n_sample)]
        self.pos_vec = [0 for i in range(n_sample)]
        self.pos = [0 for i in range(n_sample)]
        self.grades = [0 for i in range(n_sample)]
        self.trajs = [0 for i in range(n_sample)]
        for i, key in enumerate(self.data_dict):
            # self.images[i] = self.data_dict[key]["Movie"][0]  # 1フレーム目を見ている
            self.pos[i] = self.data_dict[key]["Pos"]
            self.pos_vec[i]=pos_to_move_vectors(self.pos[i])
            self.pos_images[i]=pos_to_image(self.pos[i])
            self.grades[i] = self.data_dict[key]["Grade"]
            self.trajs[i] = pos_to_td(self.pos[i],output_type="pos")
        self.IDs = list(self.data_dict.keys())
        self.img_size = img_size

    # len(SpermDataset) is len(IDs)
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, i):
        # array like -> PIL Image
        # if type(self.pos_images[i]) != Image.Image:
        #     pos_image = Image.fromarray(self.pos_images[i])
        # else:
        #     pos_image = self.pos_images[i]
        pos_image = self.pos_images[i]
        if pos_image.mode == 'RGBA':
            pos_image = pos_image.convert('RGB')

        # array like -> torch.Tensor(dtype=float32)
        grade = torch.Tensor(self.grades[i])
        # pos = torch.Tensor(self.pos[i])
        pos_vec =torch.Tensor(self.pos_vec[i])        

        # PIL Image -> torch.Tensor(dtype=float32) & Resize by img_size
        img_size = self.img_size
        transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.RandomRotation(degrees=[-180, 180]),transforms.ToTensor()]
        )
        pos_image = transform(pos_image).to(torch.float32)
        
        # array like -> torch.Tensor(dtype=float32)
        traj = torch.Tensor(np.array(self.trajs[i]))  # 1次元Tensorに変換

        # str (No Use for Prediction)
        ID = str(self.IDs[i])

        return (
            pos_image,
            pos_vec,
            grade,
            ID,
        )  # {'ID' : ID, 'image' : image, 'traj' : traj, 'grade' : grade}


def pos_to_td(pos_array, output_type=None):
    move_vectors = np.array(
        [pos_array[i + 1] - pos_array[i] for i in range(len(pos_array) - 1)]
    )
    move_pos=np.array([pos_array[i] - pos_array[0] for i in range(len(pos_array))])
    norm_move_vectors = np.array([np.sqrt(x**2 + y**2) for x, y in move_vectors])

    if output_type=='speed':
        return sum(norm_move_vectors) / len(norm_move_vectors)
    elif output_type=='pos':
        return move_pos
    else:
        return sum(norm_move_vectors)

def pos_to_move_vectors(pos):
    x_min=9999
    y_min=9999
    for x,y in pos:
      if x < x_min:
        x_min=x
      if y < y_min:
        y_min=y
    return np.array([pos[i] - [x_min,y_min] for i in range(len(pos))])

def pos_to_image(pos):
    """
    pos to PIL.Image
    """    
    move_vectors=pos_to_move_vectors(pos)
    
    x = [point[0] for point in move_vectors]
    y = [point[1] for point in move_vectors]

    # 軌跡をプロット
    fig, g = plt.subplots(figsize=(4, 4), dpi=56)  # FigureのサイズとDPIを指定
    g.plot(x, y, marker='o', linestyle='-', linewidth=20)
    g.set_xlim(0, 110)
    g.set_ylim(0, 110)
    g.set_aspect('equal')
    g.axis("off")

    # Figureを保存
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=56)  # figオブジェクトを使用してDPIを指定して保存
    plt.close(fig)
    buf.seek(0)
    
    # PIL Imageオブジェクトを作成
    image = Image.open(buf)
    return image


def make_kfold_dataloader(args, dataset):
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    num_gpu = args.num_gpu
    if args.dist:  # distributed training
        batch_size = args.batch_size
        num_workers = args.num_worker
    else:  # non-distributed training
        multiplier = 1 if num_gpu == 0 else num_gpu
        batch_size = args.batch_size * multiplier
        num_workers = args.num_worker*multiplier
    # 分割結果のデータの保存
    kfold_id_train = []  # trainIDの各分割結果
    kfold_id_test = []  # testIDの各分割結果
    kfold_train_dataloader_learn = (
        []
    )  # train dataloader for learning (batch size non 1)
    kfold_test_dataloader = []  # test dataloader
    splitter = GroupKFold(n_splits=args.kfold)
    # split = splitter.split(dataset, groups=dataset['Id'])
    # print("#########################################")
    # print(splitter.split(dataset, groups=dataset.IDs))
    # for train_indices, val_indices in splitter.split(dataset, groups=dataset.IDs):
    for train_indices, val_indices in kf.split(dataset):

        # prepare date
        train_dataset, val_dataset = train_test_split_torch_designated(
            dataset, train_indices, val_indices
        )
        train_dataloader_learn = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        ## dataloader
        kfold_train_dataloader_learn.append(train_dataloader_learn)
        kfold_test_dataloader.append(test_dataloader)

    return kfold_train_dataloader_learn, kfold_test_dataloader

class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_test_split_torch_designated(dataset, train_indice, test_indice):
    return Subset(dataset, train_indice), Subset(dataset, test_indice)

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# def LoadData(args):

#     with open(F"{args.data_dir}/{args.grade_path}", mode="rb") as f:
#         grade_dict = pickle.load(f)    
#     with open(F"{args.data_dir}/{args.video_path}", mode="rb") as f:
#         video_dict = pickle.load(f)
#     with open(F"{args.data_dir}/{args.traj_path}", mode="rb") as f:
#         traj_dict = pickle.load(f)
    
#     sperm_ids = np.array(list(grade_dict.keys()))

#     log.info(f'[DATA] (grade) {len(grade_dict)}    (video) {len(video_dict)}    (traj) {len(traj_dict)}')
#     return sperm_ids, grade_dict, video_dict, traj_dict

# class PackPathway(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.slowfast_alpha = 4
        
#     def forward(self, frames: torch.Tensor):
#         fast_pathway = frames
#         slow_pathway = torch.index_select(
#             frames,
#             1,
#             torch.linspace(
#                 0, frames.shape[1] - 1, frames.shape[1]//self.slowfast_alpha
#             ).long(),
#         )
#         frame_list = [slow_pathway, fast_pathway]
#         return frame_list

# #データセット作成
# class SpermDataset(Dataset):
#     # structure of SpermDataset
#     def __init__(self, videos, trajs, grades, IDs, num_frame, video_size=150, pathway=False):
#         self.videos = videos
#         self.grades = grades
#         self.trajs = trajs
#         self.IDs = IDs
#         self.num_frame = num_frame
#         self.video_size = video_size
#         self.pathway = pathway

#     def __len__(self):
#         return len(self.IDs)

#     def __getitem__(self, i):
#         mean = [0.45, 0.45, 0.45]
#         std = [0.225, 0.225, 0.225]
#         if self.pathway != 'slowfast':
#             transform = Compose(
#                 [
#                     UniformTemporalSubsample(self.num_frame)
#                     Lambda(lambda x: x/255.0),
#                     NormalizeVideo(mean, std),
#                     ShortSideScale(
#                         size=self.video_size
#                     )
#                 ]
#             )
        
#         else:
#             transform = Compose(
#                 [
#                     UniformTemporalSubsample(self.num_frame),
#                     Lambda(lambda x: x/255.0),
#                     NormalizeVideo(mean, std),
#                     ShortSideScale(
#                         size=self.video_size
#                     ),
#                     PackPathway()
#                 ]
#             ) 
            
#         video = torch.from_numpy(self.videos[i].astype(np.float32)).permute(3,0,1,2)
#         video = transform(video)

#         grade = torch.Tensor(self.grades[i])
#         traj = torch.Tensor(np.array(self.trajs[i]))
#         ID = str(self.IDs[i])
#         return video, traj, grade, ID

# def make_kfold_dataloader(k, sperm_ids, video_dict, traj_dict, grade_dict, num_frame=8, video_size=224, batch_size=32, pathway=False):
#     kf = KFold(n_splits=k, shuffle=True, random_state=123)

#     kfold_id_train = []          
#     kfold_id_test = []            
#     kfold_train_dataloader_learn = []  
#     kfold_test_dataloader = []        

#     for index_train, index_test in kf.split(sperm_ids):
#         train_id = sperm_ids[index_train]
#         test_id = sperm_ids[index_test]

#         train_video = np.array([video_dict[ID] for ID in train_id])
#         train_traj = np.array([traj_dict[ID] for ID in train_id])

#         train_grade = np.array([grade_dict[ID] for ID in train_id])

#         test_video = np.array([video_dict[ID] for ID in test_id])
#         test_traj = np.array([traj_dict[ID] for ID in test_id])
     
#         test_grade = np.array([grade_dict[ID] for ID in test_id])

#         train_dataset = SpermDataset(videos=train_video, trajs=train_traj, grades=train_grade, IDs=train_id, num_frame=num_frame, video_size=video_size, pathway=pathway)
#         test_dataset = SpermDataset(videos=test_video, trajs=test_traj, grades=test_grade, IDs=test_id, num_frame=num_frame, video_size=video_size, pathway=pathway)

#         train_dataloader_learn = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
#         kfold_train_dataloader_learn.append(train_dataloader_learn)
#         kfold_test_dataloader.append(test_dataloader)

#     return kfold_train_dataloader_learn, kfold_test_dataloader