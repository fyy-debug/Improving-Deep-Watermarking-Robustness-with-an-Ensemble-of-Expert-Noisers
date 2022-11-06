import os
import csv
import gc
import numpy as np
import torch
from absl import flags, app
from torchvision import datasets, transforms
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.necst2 import NECST
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from model.vit_mask import ViT
import utils
from losses import loss_map
from noise_layers.noiser import Noiser
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.dropout import Dropout
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.hue import Hue
from noise_layers.gaussian_noise import Gaussian_Noise
from noise_layers.sat import Sat
from noise_layers.blur import Blur
from torchjpeg.metrics import psnr as PNSR
from noise_layers.jpeg import JPEG
from model.cnn_mask import CNN_MASK

###########################
FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'exp', 'experiments name')
flags.DEFINE_string('resume', None, 'resume from checkpoint')
flags.DEFINE_string('resume_pretrain', None, 'resume from pretrain checkpoint')
flags.DEFINE_string('dataset', '/home/yeochengyu/Documents/flickr', 'Dataset used')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_integer('image_size', 128, "size of the images generated")
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_integer('nc', 3, "Channel Dimension of image")
flags.DEFINE_integer('seed', 1, "random seed")
flags.DEFINE_bool('cuda', True, 'Flag using GPU')
flags.DEFINE_integer('pretrain_iter', 5000, 'iterations of HiDDeN identity pretrain')
flags.DEFINE_integer('eval_every', 500, "validation bit error with every this epoch ")
flags.DEFINE_integer('print_every', 50, "print training information with every this epoch")
flags.DEFINE_integer('save_every', 50, "saving checkpoint with every this epoch")
flags.DEFINE_integer('redundant_length', 120, "length of redundant message")
flags.DEFINE_integer('message_length', 30, "length of message")
flags.DEFINE_integer('iter', 1000000, "number of iterations model trained")

### NECST ###
flags.DEFINE_integer('necst_iter', 20000, "number of iterations necst trained")
### Decoder ###
flags.DEFINE_integer('decoder_channels', 64, "number of channels of decoder")
flags.DEFINE_integer('decoder_blocks', 7, "number of blocks of decoder")
### Encoder ###
flags.DEFINE_integer('encoder_channels', 64, "number of channels of encoder")
flags.DEFINE_integer('encoder_blocks', 4, "number of blocks of encoder")
### Discriminator ###
flags.DEFINE_integer('discriminator_channels', 64, "number of channels of discriminator")
flags.DEFINE_integer('discriminator_blocks', 3, "number of blocks of discriminator")
flags.DEFINE_enum('loss', 'ns', loss_map.keys(), "loss function")
### Generator ###
flags.DEFINE_integer('n_gen', 5, "number of generator train per iterations")
flags.DEFINE_integer('local_rank', 0, "rank of GPU used")


def train():
    device = torch.device("cuda", FLAGS.local_rank)
    print("device_name :", torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("using CPU")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            # transforms.CenterCrop((FLAGS.image_size, FLAGS.image_size)),
            transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    dist.init_process_group(backend='nccl')
    dist.barrier()
    #     rank = dist.get_rank()
    world_size = dist.get_world_size()


    train_images = datasets.ImageFolder(FLAGS.dataset + "/train", data_transforms['train'])
    validation_images = datasets.ImageFolder(FLAGS.dataset + "/val", data_transforms['test'])

    train_sampler = DistributedSampler(train_images)
    valid_sampler = DistributedSampler(validation_images)


    train_loader = torch.utils.data.DataLoader(train_images, sampler=train_sampler, batch_size=FLAGS.batch_size,pin_memory=False, prefetch_factor=2, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_images, sampler=valid_sampler, batch_size=FLAGS.batch_size,pin_memory=False, prefetch_factor=2, num_workers=4)

    file_count = len(train_loader.dataset)
    if file_count % FLAGS.batch_size == 0:
        FLAGS.print_every = file_count // FLAGS.batch_size
    else:
        FLAGS.print_every = file_count // FLAGS.batch_size + 1

    FLAGS.save_every = FLAGS.print_every * FLAGS.save_every
    FLAGS.eval_every = FLAGS.print_every
    print("training file_count:", file_count)
    print("Print information every {} step".format(FLAGS.print_every))

    looper = utils.infiniteloop(train_loader, FLAGS.message_length, device)

    ### Model ###
    net_Encdec = EncoderDecoder(FLAGS).to(device)
    net_Dis = Discriminator(FLAGS).to(device)

    net_Gen = ViT(
        image_size=FLAGS.image_size,
        patch_size=8,
        dim=256,
        depth=6,
        heads=12,
        mlp_dim=256,
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)

    net_Encdec = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_Encdec)
    net_Encdec = torch.nn.parallel.DistributedDataParallel(net_Encdec, device_ids=[FLAGS.local_rank],output_device=FLAGS.local_rank,broadcast_buffers=False)

    net_Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_Dis)
    net_Dis = torch.nn.parallel.DistributedDataParallel(net_Dis, device_ids=[FLAGS.local_rank],output_device=FLAGS.local_rank,broadcast_buffers=False)

    net_Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_Gen)
    net_Gen = torch.nn.parallel.DistributedDataParallel(net_Gen, device_ids=[FLAGS.local_rank],output_device=FLAGS.local_rank,broadcast_buffers=False,find_unused_parameters=True)

    net_Necst = NECST(FLAGS, device).to(device)
    net_Necst = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_Necst).to(device)
    net_Necst = torch.nn.parallel.DistributedDataParallel(net_Necst, device_ids=[FLAGS.local_rank],output_device=FLAGS.local_rank,broadcast_buffers=False)

    noiser = Noiser(device)

    ### Optimizer ###
    optim_EncDec = torch.optim.Adam(net_Encdec.parameters())
    optim_Dis = torch.optim.Adam(net_Dis.parameters())
    optim_Gen = torch.optim.Adam(net_Gen.parameters(), lr=1e-4, weight_decay=1e-3)

    ## Loading checkpoint ##
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        net_Encdec.load_state_dict(checkpoint['enc-dec-model'])
        optim_EncDec.load_state_dict(checkpoint['enc-dec-optim'])
        net_Dis.load_state_dict(checkpoint['discrim-model'])
        optim_Dis.load_state_dict(checkpoint['discrim-optim'])
        net_Gen.load_state_dict(checkpoint['generator-model'])
        optim_Gen.load_state_dict(checkpoint['generator-optim'])
        net_Necst.load_state_dict(checkpoint['necst'])
        start = int(checkpoint['iter']) + 1
        epoch = int(checkpoint['epoch']) + 1
        FLAGS.out_dir = FLAGS.resume.split("checkpoint")[0]
        print("Loaded checkpoint from iter:{} / Epoch:{} in dir{}".format(start, epoch, FLAGS.out_dir))
        del checkpoint

    elif FLAGS.resume_pretrain:
        start = 1
        epoch = 0
        checkpoint = torch.load(FLAGS.resume_pretrain)
        net_Encdec.load_state_dict(checkpoint['enc-dec-model'])
        optim_EncDec.load_state_dict(checkpoint['enc-dec-optim'])
        net_Dis.load_state_dict(checkpoint['discrim-model'])
        optim_Dis.load_state_dict(checkpoint['discrim-optim'])
        print("Loaded pretrained checkpoint from ", FLAGS.resume_pretrain)
        print("output dir :", FLAGS.out_dir)
        del checkpoint
    else:
        start = 1
        epoch = 0
        print("output dir :", FLAGS.out_dir)

    ## Define loss functions ##
    dis_loss = loss_map[FLAGS.loss]
    mse_loss = torch.nn.MSELoss()

    ## Pretraining with HiDDeN identity ##
    if FLAGS.pretrain_iter > 0 and FLAGS.resume_pretrain is None and FLAGS.resume is None:
        for i in range(1, FLAGS.pretrain_iter):
            ########### training discriminator ###############
            optim_Dis.zero_grad()
            images, _ = next(looper)
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], FLAGS.redundant_length))).to(device)
            pred_real = net_Dis(images)
            encoded_images, decoded_messages = net_Encdec(images, messages, net_Gen, net_Necst, identity=True)
            pred_fake = net_Dis(encoded_images.detach())
            loss_fake, loss_real = dis_loss(pred_fake, pred_real)
            loss_D = (loss_real + 0.5 * loss_fake) * 1.0
            loss_D.backward()
            optim_Dis.step()

            ########### training Encoder Decoder ###############
            optim_EncDec.zero_grad()
            pred_fake = net_Dis(encoded_images)
            loss_fake = dis_loss(pred_fake)
            enc_dec_image_loss = mse_loss(encoded_images, images)
            enc_dec_message_encloss = mse_loss(decoded_messages, messages)
            loss_ED = 0.15 * loss_fake + 1.0 * enc_dec_image_loss + 5.0 * enc_dec_message_encloss
            loss_ED.backward()
            optim_EncDec.step()

            if (i == 1 or i % FLAGS.print_every == 0) and FLAGS.local_rank is 0:
                net_Encdec.eval()
                net_Dis.eval()
                with torch.no_grad():
                    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                    bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                            messages.shape[0] * messages.shape[1])).item()
                    pnsr = PNSR((encoded_images + 1) / 2, (images + 1) / 2).mean().item()
                    print("\n########## LOCAL RANK{} Iteration:{} ##########\n".format(FLAGS.local_rank,i))
                    print("pnsr                   :", pnsr)
                    print("loss_Discim            :", loss_D.item())
                    print("loss_Encdec            :", loss_ED.item())
                    print("encdec_message_encloss :", enc_dec_message_encloss.item())
                    print("encdec_image_loss      :", enc_dec_image_loss.item())
                    print("bitwise_avg_err        :", bitwise_avg_err)
                    print("\n###################################\n")

                net_Encdec.train()
                net_Dis.train()

        checkpoint = {
            'enc-dec-model': net_Encdec.state_dict(),
            'enc-dec-optim': optim_EncDec.state_dict(),
            'discrim-model': net_Dis.state_dict(),
            'discrim-optim': optim_Dis.state_dict(),
            'iter': i
        }
        torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "pretrain_iter{}.pyt".format(i))
        print('iter {} Saving pretrain checkpoint done.'.format(i))

    if FLAGS.necst_iter > 0 and FLAGS.resume is None:
        net_Necst.module.pretrain()
    net_Necst.module.eval()

    ##### Start Training ####

    for i in range(start, FLAGS.iter):

        ########### training generator ###############
        images, messages = next(looper)
        redundant_messages = net_Necst.module.encode(messages).detach()
        encoded_images = net_Encdec.module.encoder(images, redundant_messages).detach()

        for step in range(FLAGS.n_gen):
            optim_Gen.zero_grad()
            adv_images = net_Gen(encoded_images)  ##
            adv_dec_messages = net_Encdec.module.decoder(adv_images)
            gen_image_loss = mse_loss(adv_images, encoded_images)
            gen_message_loss = mse_loss(adv_dec_messages, redundant_messages)
            loss_G = 15.0 * gen_image_loss - 1.0 * gen_message_loss
            loss_G.backward()
            optim_Gen.step()

        ########### training discriminator ###############
        optim_Dis.zero_grad()
        pred_real = net_Dis(images)
        encoded_images, adv_images, redundant_decoded_messages, adv_redundant_decoded_messages = net_Encdec(images,
                                                                                                            messages,
                                                                                                            net_Gen.module,
                                                                                                            net_Necst.module)
        pred_fake = net_Dis(encoded_images.detach())
        loss_fake, loss_real = dis_loss(pred_fake, pred_real)
        loss_D = (1.0 * loss_real + 0.5 * loss_fake) * 1.0
        loss_D.backward()
        optim_Dis.step()

        ########### training Encoder Decoder ###############
        optim_EncDec.zero_grad()
        pred_fake = net_Dis(encoded_images)
        loss_fake = 1.0 * dis_loss(pred_fake)
        enc_dec_image_loss = mse_loss(encoded_images, images)
        enc_dec_message_encloss = mse_loss(redundant_decoded_messages, redundant_messages)
        enc_dec_message_advloss = mse_loss(adv_redundant_decoded_messages, redundant_messages)
        loss_ED = 0.01 * loss_fake + 3.5 * enc_dec_image_loss + 0.3 * enc_dec_message_encloss + 0.2 * enc_dec_message_advloss
        loss_ED.backward()
        optim_EncDec.step()

        if (i == 1 or i % FLAGS.print_every == 0 )and FLAGS.local_rank is 0:
            net_Encdec.eval()
            net_Necst.eval()
            net_Gen.eval()
            net_Dis.eval()
            with torch.no_grad():
                decoded_messages = net_Necst.module.decode(redundant_decoded_messages)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                        messages.shape[0] * messages.shape[1])).item()
                losses_accu = {
                    'loss_Gen                ': loss_G.item(),
                    'loss_Discrim            ': loss_D.item(),
                    'loss_Encdec             ': loss_ED.item(),
                    'gen_image_loss          ': gen_image_loss.item(),
                    'gen_message_loss        ': gen_message_loss.item(),
                    'enc_dec_message_encloss ': enc_dec_message_encloss.item(),
                    'enc_dec_message_advloss ': enc_dec_message_advloss.item(),
                    'encdec_image_loss       ': enc_dec_image_loss.item(),
                    'bitwise_avg_err         ': bitwise_avg_err}

                print("\n########## Epoch:{} Training ##########\n".format(epoch))
                for key, value in losses_accu.items():
                    print(key + ":", value)
                print("\n###################################\n")

                with open(FLAGS.out_dir + "/train.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if i == 1:
                        row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
                        writer.writerow(row_to_write)
                    row_to_write = [epoch] + ['{:.4f}'.format(loss_avg) for loss_avg in losses_accu.values()]
                    writer.writerow(row_to_write)

            torch.cuda.empty_cache()
            net_Encdec.train()
            net_Necst.train()
            net_Gen.train()
            net_Dis.train()

        if (i == 1 or i % FLAGS.eval_every == 0) and FLAGS.local_rank is 0:
            losses_accu = {
                'Identity_err': 0.00,
                'Crop_err': 0.00,
                'Cropout_err': 0.00,
                'Dropout_err': 0.00,
                'Jpeg_err': 0.00,
                'Resize_err': 0.00,
                'Gaussian_err': 0.00,
                'Blur_err': 0.00,
                'Sat_err': 0.00,
                'Hue_err': 0.00,
                'PNSR': 0.00}
            count = 0
            net_Encdec.eval()
            net_Necst.eval()
            net_Gen.eval()
            net_Dis.eval()
            with torch.no_grad():
                for images, _ in validation_loader:
                    count += 1
                    images = images.to(device)
                    messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], FLAGS.message_length))).to(
                        device)
                    for noise in noiser.noise_layers:
                        redundant_messages = net_Necst.module.encode(messages)
                        encoded_images = net_Encdec.module.encoder(images, redundant_messages)
                        noised_and_cover = noise([encoded_images, images])
                        noised_images = noised_and_cover[0]
                        redundant_decoded_messages = net_Encdec.module.decoder(noised_images)
                        decoded_messages = net_Necst.module.decode(redundant_decoded_messages)
                        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                        bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                                    messages.shape[0] * messages.shape[1])).item()

                        if isinstance(noise, Identity):
                            losses_accu['Identity_err'] += bitwise_avg_err


                        elif isinstance(noise, Crop):
                            losses_accu['Crop_err'] += bitwise_avg_err

                        elif isinstance(noise, Cropout):
                            losses_accu['Cropout_err'] += bitwise_avg_err

                        elif isinstance(noise, Dropout):
                            losses_accu['Dropout_err'] += bitwise_avg_err

                        elif isinstance(noise, JPEG):
                            losses_accu['Jpeg_err'] += bitwise_avg_err

                        elif isinstance(noise, Resize):
                            losses_accu['Resize_err'] += bitwise_avg_err

                        elif isinstance(noise, Hue):
                            losses_accu['Hue_err'] += bitwise_avg_err

                        elif isinstance(noise, Blur):
                            losses_accu['Blur_err'] += bitwise_avg_err

                        elif isinstance(noise, Gaussian_Noise):
                            losses_accu['Gaussian_err'] += bitwise_avg_err

                        elif isinstance(noise, Sat):
                            losses_accu['Sat_err'] += bitwise_avg_err

            losses_accu['Identity_err'] /= count
            losses_accu['Crop_err'] /= count
            losses_accu['Cropout_err'] /= count
            losses_accu['Dropout_err'] /= count
            losses_accu['Jpeg_err'] /= count
            losses_accu['Resize_err'] /= count
            losses_accu['Hue_err'] /= count
            losses_accu['Blur_err'] /= count
            losses_accu['Gaussian_err'] /= count
            losses_accu['Sat_err'] /= count

            encoded_images = net_Encdec.module.encoder(images, redundant_messages)
            losses_accu['PNSR'] = PNSR((encoded_images + 1) / 2, (images + 1) / 2).mean().item()
            adv_images = net_Gen(encoded_images)

            utils.save_images(images.cpu()[:8, :, :, :],
                              encoded_images[:8, :, :, :].cpu(),
                              epoch,
                              os.path.join(FLAGS.out_dir, 'images'), resize_to=(128, 128), imgtype="enc")

            utils.save_images(images.cpu()[:8, :, :, :],
                              adv_images[:8, :, :, :].cpu(),
                              epoch,
                              os.path.join(FLAGS.out_dir, 'images'), resize_to=(128, 128), imgtype="adv")

            print("\n##########LOCAL_RANK{} Epoch:{} Validation ##########\n".format(FLAGS.local_rank,epoch))
            print("Identity_err     :", round(losses_accu['Identity_err'], 6))
            print("Crop_err         :", round(losses_accu['Crop_err'], 6))
            print("Cropout_err      :", round(losses_accu['Cropout_err'], 6))
            print("Dropout_err      :", round(losses_accu['Dropout_err'], 6))
            print("Jpeg_err         :", round(losses_accu['Jpeg_err'], 6))
            print("Resize_err       :", round(losses_accu['Resize_err'], 6))
            print("Gaussian_err     :", round(losses_accu['Gaussian_err'], 6))
            print("Blur_err         :", round(losses_accu['Blur_err'], 6))
            print("Sat_err          :", round(losses_accu['Sat_err'], 6))
            print("Hue_err          :", round(losses_accu['Hue_err'], 6))
            print("PNSR             :", round(losses_accu['PNSR'], 6))

            print("Count            :", count)
            print("\n###################################\n")

            with open(FLAGS.out_dir + "/validation.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if i == 1:
                    row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
                    writer.writerow(row_to_write)
                row_to_write = [epoch] + ['{:.4f}'.format(loss_avg) for loss_avg in losses_accu.values()]
                writer.writerow(row_to_write)

            epoch += 1
            del losses_accu
            gc.collect()
            torch.cuda.empty_cache()
            #train_sampler.set_epoch(epoch)
            #valid_sampler.set_epoch(epoch)
            net_Encdec.train()
            net_Necst.train()
            net_Gen.train()
            net_Dis.train()

        if (i == 1 or i % FLAGS.save_every == 0) and FLAGS.local_rank is 0:
            ## saving checkpoint ##
            checkpoint = {
                'enc-dec-model': net_Encdec.state_dict(),
                'enc-dec-optim': optim_EncDec.state_dict(),
                'discrim-model': net_Dis.state_dict(),
                'discrim-optim': optim_Dis.state_dict(),
                'generator-model': net_Gen.state_dict(),
                'generator-optim': optim_Gen.state_dict(),
                'necst': net_Necst.state_dict(),
                'iter': i,
                'epoch': epoch - 1
            }
            torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "epoch{}.pyt".format(epoch - 1))
            print('Epoch {} Saving checkpoint done.'.format(epoch - 1))
            del checkpoint
            gc.collect()


def main(argv):
    # utils.set_seed(FLAGS.seed)

    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    run = 0
    while os.path.exists(FLAGS.out_dir + FLAGS.name + str(run) + "/"):
        run += 1
    FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + str(run) + "/"
    if FLAGS.local_rank is 0:
        os.mkdir(FLAGS.out_dir)
        os.mkdir(FLAGS.out_dir + "images")
        os.mkdir(FLAGS.out_dir + "checkpoint")
    train()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

## Command ##
## train from scratch (not pretraining )---> python train.py  --batch_size 32 --dataset /flickr --pretrain_iter 0
## train from scratch (pretraining)     ---> python train.py  --batch_size 32 --dataset /flickr --pretrain_iter 5000
## load pretrain checkpoint ##          ---> python train.py  --batch_size 32 --resume_pretrain /checkpoint/pretrain500.pyt
## load train checkpoint ##             ---> python train.py  --batch_size 32 --resume /checkpoint/train500.pyt --> auto skip necst pretraining

