import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Sperm Assessment")
    parser.add_argument('--name', default='01_model_framesetting')
    ########## base options ##########
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--data_dir', default='dataset')
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--grade_path', default='grades.pkl')
    parser.add_argument('--video_path', default='16frame.pkl')
    parser.add_argument('--traj_path', default='td.pkl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default='234', type=int)
    parser.add_argument('--wandb_project',default='OFS_grade_estimate')
    ########## data options ##########
    parser.add_argument(
        "--sperm_data_path", default="/home/kodama/OFS_grade_estimate/dataset/SpermDataBase.pkl", type=str
    )
    parser.add_argument('--num_frame', default=16, type=int, help='default 16, 8 if timesformer, 32 if slowfast')
    parser.add_argument('--video_size', default=224, type=int)
    ########## model options ##########
    parser.add_argument('--model_name', default='timesformer', choices=['timesformer', 'rostfine', 'slowfast', 'r3d', 'x3d', 'r21d', 'i3d', 'vivit', 'vgg'])
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--pretrained_timesformer', default='/home/kodama/OFS_grade_estimate/models/TimeSformer_divST_8x32_224_K400.pyth')
    parser.add_argument('--pretrained_vivit', default='/home/kodama/OFS_grade_estimate/art/weight/vivit_model.pth')
    parser.add_argument('--pretrained_dir', default='facebookresearch/pytorchvideo')
    parser.add_argument('--frame_select',default='zero',choices=['None','zero','random','all','ofs'])
    ########## train options ##########
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--js', default=False, type=bool)
    ########## test options ##########
    parser.add_argument('--plot_w', default=16, type=int)
    parser.add_argument('--plot_h', default=5, type=int)
    parser.add_argument('--load_dir', default='output/model_namme/time')
    parser.add_argument('--use_metrics', default='mse+js+balanced_acc')
    ########## RoSTFine options ##########
    parser.add_argument('--use_div', default='gs+gt+st')
    parser.add_argument('--use_feat', default='vg+vs+vt')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_blk', default=2, type=int)
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--top_attn', default=11, type=int)
    #Distribute learning setting
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_worker',type=int, default=6)

    args = parser.parse_args()
    return args