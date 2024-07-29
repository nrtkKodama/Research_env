import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision.models import vgg16, resnet18, resnet50

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  # initialize the process group
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)
  def forward(self, x):
    return self.net2(self.relu(self.net1(x)))

class MyVgg(nn.Module):
  def __init__(self):
      super(MyVgg, self).__init__()        
      self.features = vgg16(pretrained=True).features
      self.avgpool = vgg16(pretrained=True).avgpool
      self.flatten_map = nn.Sequential(
          nn.Flatten(start_dim=1),
          nn.Linear(in_features=512*7*7, out_features=512, bias=True),
          nn.ReLU(inplace=True),
      )
      self.classifier = nn.Sequential(
          nn.Linear(in_features=512, out_features=5, bias=True),
          nn.Softmax(dim=1)
      )

  def forward(self, img):
      img = self.features(img)
      img = self.avgpool(img)
      img = self.flatten_map(img)
      out = self.classifier(img)
      return out


def demo_basic(rank, world_size):
  print(f"Running basic DDP example on rank {rank}.")
  setup(rank, world_size)

  # create model and move it to GPU with id rank
  # model = ToyModel().to(rank)
  model = MyVgg().to(rank)

  ddp_model = DDP(model, device_ids=[rank])

  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

  optimizer.zero_grad()
  outputs = ddp_model(torch.randn(20, 10))
  labels = torch.randn(20, 5).to(rank)
  loss_fn(outputs, labels).backward()
  optimizer.step()

  cleanup()


def run_demo(demo_fn, world_size):
  mp.spawn(demo_fn,
    args=(world_size,),
    nprocs=world_size,
    join=True)

if __name__ == "__main__":
    run_demo(demo_basic, 3)