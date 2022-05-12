import warnings
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from modules.env import AttrDict, build_env
from src.preprocessing.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models.hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.layers.losses import feature_loss, generator_loss, discriminator_loss
from src.layers.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

warnings.simplefilter(action='ignore', category=FutureWarning)
torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    global start_b
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        # print(generator)
        os.makedirs(a.save_path, exist_ok=True)
        print("model save directory: ", a.save_path)

    if os.path.isdir(a.pretrained_checkpoint):
        cp_g = scan_checkpoint(a.pretrained_checkpoint, 'g_')
        cp_do = scan_checkpoint(a.pretrained_checkpoint, 'do_')
    else:
        cp_g, cp_do = None, None

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        # steps = state_dict_do['steps'] + 1
        # last_epoch = state_dict_do['epoch']

    last_epoch = -1

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.save_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc='Epoch {}'.format(epoch + 1))
        for i, batch in enumerate(pbar):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch

            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            # generator audio from mel-spectrogram
            y_g_hat = generator(x)

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                          h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)
            optim_d.zero_grad()

            # Calculate loss between input audio and generated audio
            # MPD multi-scale discriminator
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            # print('MSD working...')
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # Calculate all loss
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss (lambda mel = 45)
            # Mel-spectrogram loss: Calculate L1 distance between mel from input audio and mel from generated audio
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            # Calculate loss from Discriminator and Generator again => total Generator loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            # Discriminator feature loss of input audio and generated audio
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            # Generator loss calculated by Dicriminator loss
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            # Last Generator loss is sum of all loss
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            with torch.no_grad():
                mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            pbar.set_postfix({'steps': '{:d}'.format(steps), 'generator losses': '{:4.3f}'.format(loss_gen_all),
                              'mel-spectrogram losses': '{:4.3f}'.format(mel_error),
                              'time per step': ' {:4.3f}s'.format(time.time() - start_b)})
            steps += 1

        # checkpointing
        # if steps % a.checkpoint_interval == 0 and steps != 0:
        save_path = "{}/g_{:08d}".format(a.save_path, steps)
        save_checkpoint(save_path,
                        {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

        if steps > 2500000:
            save_path = "{}/do_{:08d}".format(a.save_path, steps)
            save_checkpoint(save_path,
                            {'mpd': (mpd.module if h.num_gpus > 1
                                     else mpd).state_dict(),
                             'msd': (msd.module if h.num_gpus > 1
                                     else msd).state_dict(),
                             'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                             'epoch': epoch})
            exit()
        else:
            save_path = "{}/do_current".format(a.save_path)
            save_checkpoint(save_path,
                            {'mpd': (mpd.module if h.num_gpus > 1
                                     else mpd).state_dict(),
                             'msd': (msd.module if h.num_gpus > 1
                                     else msd).state_dict(),
                             'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                             'epoch': epoch})

        # Tensorboard summary logging
        # if steps % a.summary_interval == 0:
        sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
        sw.add_scalar("training/mel_spec_error", mel_error, steps)

        if a.train_pretrain_model:
            continue
        # Validation
        print('Evaluate validation data...')
        generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0
        with torch.no_grad():
            for j, batch in enumerate(validation_loader):
                x, y, _, y_mel = batch
                y_g_hat = generator(x.to(device))
                y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                              h.hop_size, h.win_size,
                                              h.fmin, h.fmax_for_loss)
                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                if j <= 4:
                    if steps == 0:
                        sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                        sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                    sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)

                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                 h.hop_size, h.win_size,
                                                 h.fmin, h.fmax_for_loss)
                    sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                  plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

            val_err = val_err_tot / (j + 1)
            sw.add_scalar("validation/mel_spec_error", val_err, steps)
        print('Validation loss: {:4.3f} \nValidation mel-spec error: {:4.3f}'.format(val_err_tot, val_err))
        if epoch == 0:
            best_val_err_tot = val_err_tot
            patient = 0
            print('Save for 1st epoch!!!')
            os.makedirs("{}/best_model".format(a.save_path), exist_ok=True)
            save_path = "{}/best_model/g_{:08d}".format(a.save_path, steps)
            save_checkpoint(save_path,
                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
        else:
            if val_err_tot < best_val_err_tot:
                best_val_err_tot = val_err_tot
                patient = 0
                print('Best loss so far!')
                save_path = "{}/best_model/g_{:08d}".format(a.save_path, steps)
                save_checkpoint(save_path,
                                {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
            else:
                patient += 1
                print('Not good epoch!!! - patient: {}'.format(patient))

        generator.train()
        scheduler_g.step()
        scheduler_d.step()


def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='dataset/wavs')
    parser.add_argument('--input_mels_dir', default='dataset/mels')
    parser.add_argument('--input_training_file', default='data/train_files.txt')
    parser.add_argument('--input_validation_file', default='data/test_files.txt')
    parser.add_argument('--pretrained_checkpoint', default='')
    parser.add_argument('--save_path', default='saved_models/checkpoint')
    parser.add_argument('--config', default='', required=True)
    parser.add_argument('--training_epochs', default=200, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--multi_gpus', default=False, type=bool)
    parser.add_argument('--train_pretrain_model', default=False, type=bool)

    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.save_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        if a.multi_gpus:
            h.num_gpus = torch.cuda.device_count()
        else:
            h.num_gpus = 1
        print('Total GPU use: {}'.format(h.num_gpus))
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
