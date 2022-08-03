import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from module_list import compressor_list, optimizer_list
from dataset import Dataset
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

# Speed up when the model structure is fixed
torch.backends.cudnn.benchmark = True
# Required for some operators to work normally when deterministic_algorithms is enabled
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# For code decode consistence
torch.use_deterministic_algorithms(True)


def train(rank, a, h):
    # Init DDP devices
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Import model
    compressor = compressor_list(a, h, rank).to(device)

    # Print the model and the saving path
    save_path = os.path.join(a.checkpoint_path, a.model_name)
    if rank == 0:
        print(compressor)
        print("checkpoints directory : ", save_path)

    # Scan the checkpoints
    if os.path.isdir(save_path):
        com_cp = scan_checkpoint(save_path, a.model_name + '_')

    # Load the checkpoints
    if a.fine_tuning:
        if com_cp is None:
            raise Exception('No checkpoints found! Cannot finetune!')
        else:
            state_dict_com = load_checkpoint(com_cp, device)
            compressor.load_state_dict(state_dict_com['compressor'])
            steps = state_dict_com['steps'] + 1
            last_epoch = state_dict_com['epoch']
    else:
        state_dict_com = None
        steps = 0
        last_epoch = -1

    # Put the models to DDP
    if h.num_gpus > 1:
        compressor = DistributedDataParallel(compressor, device_ids=[rank], find_unused_parameters=True).to(device)

    # Init optimizer
    optim_com = optimizer_list(compressor, h)
    if a.fine_tuning and state_dict_com is not None:
        optim_com.load_state_dict(state_dict_com['optim_com'])
    scheduler_com = torch.optim.lr_scheduler.ExponentialLR(optim_com, gamma=h.lr_decay, last_epoch=last_epoch)

    # Load training set
    trainset = Dataset(a.training_dir, h, shuffle=True)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    # Load validation set
    if rank == 0:
        validset = Dataset(a.validation_dir, is_train=False)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        # Init tensorboard
        sw = SummaryWriter(os.path.join(save_path, 'logs'))

    # Start training
    compressor.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        # Use different random seed every epoch
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        # Main training programme
        for _, batch in enumerate(train_loader):

            # Load one batch images
            img = batch
            img = torch.autograd.Variable(img.to(device, non_blocking=True))

            # Calculate loss
            """loss_items = compressor(img)
            loss, bit_rate, distortion = compressor.module.loss(img, loss_items) if h.num_gpus > 1 \
                                    else compressor.loss(img, loss_items)"""
            loss, bit_rate_y, bit_rate_z, distortion, _ = compressor(img) if h.num_gpus > 1 \
                else compressor(img)

            # Optimize
            optim_com.zero_grad()
            loss.backward()
            optim_com.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print('Steps : {:d}, Bit rate (y) : {:.4f}, Bit rate (z) : {:.4f}, Distortion : {:.6f}'.
                          format(steps, bit_rate_y, bit_rate_z, distortion))

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/{}_{:08d}".format(save_path, a.model_name, steps)
                    save_checkpoint(checkpoint_path,
                                    {'compressor': (compressor.module if h.num_gpus > 1 else compressor).state_dict(),
                                     'optim_com': optim_com.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/loss", loss, steps)
                    sw.add_scalar("training/bit_rate_y", bit_rate_y, steps)
                    sw.add_scalar("training/bit_rate_z", bit_rate_z, steps)
                    sw.add_scalar("training/distortion", distortion, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    compressor.eval()
                    torch.cuda.empty_cache()
                    val_err_distortion = 0
                    val_bit_rate_y = 0
                    val_bit_rate_z = 0

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            img = batch
                            img = img.to(device, non_blocking=True)
                            rec_img, val_bit_rate_y_, val_bit_rate_z_, _, _ = compressor.module.inference(
                                img) if h.num_gpus > 1 else compressor.inference(img)

                            val_err_distortion += F.mse_loss(img, rec_img).item()
                            val_bit_rate_y += val_bit_rate_y_
                            val_bit_rate_z += val_bit_rate_z_

                        val_distortion = val_err_distortion / (j + 1)
                        val_bit_rate_y = val_bit_rate_y / (j + 1)
                        val_bit_rate_z = val_bit_rate_z / (j + 1)

                        print("\nValidation results: ")
                        print("Bit rate (y) : {:.4f}, Bit rate (z) : {:.4f}, Distortion : {:.6f}\n".
                              format(val_bit_rate_y, val_bit_rate_z, val_distortion))

                        sw.add_scalar("validation/distortion", val_distortion, steps)
                        sw.add_scalar("validation/bit_rate_y", val_bit_rate_y, steps)
                        sw.add_scalar("validation/bit_rate_z", val_bit_rate_z, steps)

                    compressor.train()

            steps += 1

        scheduler_com.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process...')

    parser = argparse.ArgumentParser()

    '''
        '--model_name': Name of the model
        '--training_dir': Training data dir
        '--validation_dir': Validation data dir
        '--checkpoint_path': Path to save your model
        '--config_file': Path of your config file
        '--training_epochs': Training epochs
        '--stdout_interval': The interval steps to log
        '--checkpoint_interval': The interval steps to save your model
        '--summary_interval': The interval steps to save your curves on tensorboard
        '--validation_interval': The interval steps to do validate
        '--fine_tuning': Finetune or not
        '--lambda_': The lambda setting for RD loss
    '''

    parser.add_argument('--model_name', default='image_compressor', type=str)
    parser.add_argument('--training_dir', default=r'E:\Datasets\vimeo\video_train', type=str)
    parser.add_argument('--validation_dir', default=r'E:\Datasets\vimeo\vimeo_test', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
    parser.add_argument('--config_file', default='./configs/config.json', type=str)
    parser.add_argument('--training_epochs', default=3000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=100, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=200, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--lambda_', default=0.0483, type=float)

    a = parser.parse_args()

    # Create the path to save your model and copy the config file to the path
    with open(a.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config_file, 'config.json', os.path.join(a.checkpoint_path, a.model_name))

    # Set the random seed and check the GPU nums
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    # Main training function
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
