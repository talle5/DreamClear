'''
 * DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation
 * Modified from PixArt-alpha by Yuang Ai
 * 13/10/2024
'''
import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
import torch.nn.functional as F

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from mmcv.runner import LogBuffer
from diffusers.models import AutoencoderKL

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataloader
from diffusion.model.builder import build_model
from diffusion.model.nets import PixArtMS, ControlPixArtMSHalfSR2Branch,SwinIR
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

from basicsr.data.paired_npz_dataset import PairedNpzDataset
from PIL import Image
from torchvision.utils import save_image
from diffusion.utils.align_color import wavelet_reconstruction

warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

@torch.no_grad()
def log_validation(model,accelerator, step, device):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    save_dir = os.path.join(config.work_dir, 'validation')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,'lq'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'swinir_output'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'diffusion_output'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'color_align'), exist_ok=True)

    hw = torch.tensor([[1024, 1024]], dtype=weight_dtype, device=device).repeat(1, 1)
    ar = torch.tensor([[1.]],dtype=weight_dtype, device=device).repeat(1, 1)
    file = args.val_image
    file_name = os.path.basename(file)
    txt_info = np.load(args.val_npz)
    txt_fea = torch.from_numpy(txt_info['caption_feature']).to(device) #1 120 4096
    txt_fea = txt_fea[None] 
    attention_mask = torch.ones(1, 1, txt_fea.shape[1]) 
    if 'attention_mask' in txt_info.keys():
        attention_mask = torch.from_numpy(txt_info['attention_mask'])[None].to(device) # 1 1 120
    attention_mask = attention_mask[None]
    img_lq = Image.open(file).convert('RGB')
    img_lq.save(os.path.join(save_dir,'lq',file_name.replace('.png','_')+str(step)+'.png'))
    img_lq = np.array(img_lq)
    img_lq = torch.tensor(img_lq, dtype=vae_dtype, device=device) / 255.0
    img_lq = img_lq.permute(2, 0, 1).unsqueeze(0).contiguous()
    img_lq  = F.interpolate(
                    img_lq,
                    size=(1024,1024),
                    mode='bicubic',
                    )
    
    img_pre = swinir(img_lq)
    img_pre = torch.clamp(img_pre, 0, 1.0)
    save_image(img_pre[0], os.path.join(save_dir,'swinir_output', file_name.replace('.png','_')+str(step)+'.png'), nrow=1, normalize=True, value_range=(0, 1))

    posterior_lq = vae.encode(2*img_lq-1.0)
    latent_lq = posterior_lq.latent_dist.mode()*config.scale_factor
    latent_lq.to(dtype=weight_dtype)
    posterior_pre = vae.encode(2*img_pre-1.0)
    latent_pre = posterior_pre.latent_dist.mode()*config.scale_factor
    latent_pre.to(dtype=weight_dtype)

    if config.sampling_alg == 'DPM':
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=attention_mask,c_lq=latent_lq,c_pre=latent_pre)
        z = torch.randn(1, 4, 1024//8, 1024//8, device=device,dtype=weight_dtype)
        null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(device)
        dpm_solver = DPMS(  model.forward_with_dpmsolver,
                            condition=txt_fea,
                            uncondition=null_y,
                            cfg_scale=4.5,
                            model_kwargs=model_kwargs)
        with accelerator.autocast():
            samples = dpm_solver.sample(
                z,
                steps=40,
                order=2,
                skip_type="time_uniform",
                method="multistep",
                )
    elif config.sampling_alg == 'iddpm':
        if config.lre:
            t_diffusion = IDDPM(str(1000))
            noise=torch.randn_like(latent_pre, device=device,dtype=weight_dtype)
            timesteps = torch.randint(config.start_point, config.start_point+1, (1,), device=latent_pre.device).long()
            z = t_diffusion.q_sample(latent_pre,timesteps,noise=noise).repeat(2, 1, 1, 1)
        else:
            z = torch.randn(1, 4, 1024//8, 1024//8, device=device,dtype=weight_dtype).repeat(2, 1, 1, 1)
        null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(device)
        model_kwargs = dict(y=torch.cat([txt_fea, null_y]), cfg_scale=4.5,
                            data_info={'img_hw': hw, 'aspect_ratio': ar},
                            mask=attention_mask, c_lq=latent_lq,c_pre=latent_pre)
        diffusion = IDDPM(str(50))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device,accelerator=accelerator
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    
    samples = vae.decode(samples.to(dtype=vae_dtype) / config.scale_factor).sample
    torch.cuda.empty_cache()
    save_image(samples[0], os.path.join(save_dir,'diffusion_output', file_name.replace('.png','_')+str(step)+'.png'), nrow=1, normalize=True, value_range=(-1, 1))
    hq = (samples+1)/2
    hq = wavelet_reconstruction(hq,img_pre)
    hq = hq.clamp(0, 1.0)
    save_image(hq[0], os.path.join(save_dir,'color_align', file_name.replace('.png','_')+str(step)+'.png'), nrow=1, normalize=True, value_range=(0, 1))

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            with torch.no_grad():
                im_lq = batch['lq']
                im_gt = batch['gt']

                im_pre = swinir(im_lq.to(dtype=vae_dtype)).detach()
                im_pre = torch.clamp(im_pre, 0, 1.0)

                im_lq = im_lq*2 - 1.0
                im_pre = im_pre*2 - 1.0
                im_gt = im_gt*2 - 1.0

                im_lq = torch.clamp(im_lq,-1.0,1.0)
                im_pre = torch.clamp(im_pre,-1.0,1.0)
                im_gt = torch.clamp(im_gt,-1.0,1.0)

                #use vae to generate latent code
                posterior_lq = vae.encode(im_lq.to(dtype=vae_dtype))
                latent_lq = posterior_lq.latent_dist.mode().detach().to(dtype=weight_dtype)
                posterior_pre = vae.encode(im_pre.to(dtype=vae_dtype))
                latent_pre = posterior_pre.latent_dist.mode().detach().to(dtype=weight_dtype)
                posterior_gt = vae.encode(im_gt.to(dtype=vae_dtype))
                latent_gt = posterior_gt.latent_dist.sample().detach().to(dtype=weight_dtype)

            latent_gt = latent_gt * config.scale_factor
            latent_lq = latent_lq * config.scale_factor
            latent_pre = latent_pre * config.scale_factor

            txt_fea = batch['txt_fea'].to(dtype=weight_dtype)
            txt_mask = batch['mask']
            data_info = batch['data_info']
            for key, value in data_info.items():
                if isinstance(value, torch.Tensor):
                    data_info[key] = value.to(dtype=weight_dtype)

            # Sample a random timestep for each image
            # bs = clean_images.shape[0]
            bs = latent_gt.shape[0]

            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=latent_gt.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                # torch.autograd.set_detect_anomaly(True)
                loss_term = train_diffusion.training_losses(model ,accelerator, latent_gt, timesteps, model_kwargs=dict(y=txt_fea, mask=txt_mask, data_info=data_info, c_lq=latent_lq,c_pre=latent_pre))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                info = f"Step/Epoch [{(epoch - 1) * len(train_dataloader) + step + 1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({data_info['img_hw'][0][0].item()}, {data_info['img_hw'][0][1].item()}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            if (global_step + 1) % 1000 == 0 and config.s3_work_dir is not None:
                logger.info(f"s3_work_dir: {config.s3_work_dir}")

            global_step += 1
            data_time_start = time.time()

            synchronize()
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.eval_sampling_steps == 0:
                    log_validation(model,accelerator, (epoch - 1) * len(train_dataloader) + step + 1, model.device)
            synchronize()

        synchronize()
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)  # file permission: 666; dir permission: 777
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work_dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--resume_lr_scheduler', action='store_true')

    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--vae_pretrained', type=str, default=None)
    parser.add_argument('--swinir_pretrained', type=str, default=None)
    parser.add_argument('--val_image', type=str, default=None)
    parser.add_argument('--val_npz', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.data_root:
        config.data_root = args.data_root
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=args.resume_optimizer,
            resume_lr_scheduler=args.resume_lr_scheduler)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 6
        config.optimizer.update({'lr': args.lr})
    
    if args.load_from is not None:
        config.load_from = args.load_from

    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=9600)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length, 'class_dropout_prob': config.class_dropout_prob}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps))
    model: PixArtMS = build_model(config.model,
                                  config.grad_checkpointing,
                                  config.get('fp32_attention', False),
                                  input_size=latent_size,
                                  learn_sigma=learn_sigma,
                                  pred_sigma=pred_sigma,
                                  **model_kwargs)

    if config.load_from is not None and args.resume_from is None:
        # load from PixArt model
        missing, unexpected = load_checkpoint(config.load_from, model)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    model: ControlPixArtMSHalfSR2Branch = ControlPixArtMSHalfSR2Branch(model, copy_blocks_num=config.copy_blocks_num).train()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info("Using fp16 training for DiT.")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info("Using bf16 training for DiT.")
    vae_dtype = torch.float32

    from diffusion.model.utils import set_grad_checkpoint
    if config.grad_checkpointing:
        set_grad_checkpoint(model, config.get('fp32_attention', False), gc_step=1)
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"T5 max token length: {config.model_max_length}")

    vae = AutoencoderKL.from_pretrained(args.vae_pretrained)
    for param in vae.parameters():
        param.requires_grad_(False)
    vae.eval()
    vae = vae.to(accelerator.device, dtype=vae_dtype)

    swinir = SwinIR( img_size= 64,patch_size= 1,in_chans= 3,embed_dim= 180,depths= [6, 6, 6, 6, 6, 6, 6, 6],num_heads= [6, 6, 6, 6, 6, 6, 6, 6],
        window_size= 8,mlp_ratio= 2,sf= 8,img_range= 1.0,upsampler= "nearest+conv",resi_connection= "1conv",unshuffle= True,unshuffle_scale= 8)
    ckpt = torch.load(args.swinir_pretrained,map_location="cpu")['state_dict']
    new_ckpt = {}
    for key, value in ckpt.items():
        new_key = key.replace('module.', '')
        new_ckpt[new_key] = value
    del ckpt
    swinir.load_state_dict(new_ckpt)
    del new_ckpt
    for param in swinir.parameters():
        param.requires_grad_(False)
    swinir.eval()
    swinir = swinir.to(accelerator.device)

    if config.full_mixed:
        model.to(device=accelerator.device,dtype=weight_dtype)
        if accelerator.mixed_precision == "fp16":
            def patch_accelerator_for_fp16_training(accelerator):
                org_unscale_grads = accelerator.scaler._unscale_grads_

                def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                    return org_unscale_grads(optimizer, inv_scale, found_inf, True)

                accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
            patch_accelerator_for_fp16_training(accelerator)
    
    if config.pretrained_ckpt is not None:
        missing, unexpected = load_checkpoint(config.pretrained_ckpt, model)
        logger.info(f'load pretrained_ckpt: {config.pretrained_ckpt}')
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # if args.local_rank == 0:
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == False: logger.info(f"freeze param: {name}")
    #
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == True: logger.info(f"trainable param: {name}")

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    dataset = PairedNpzDataset(config.opt_dataset)
    train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr,base_batch_size=128)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        if args.resume_optimizer == False or args.resume_lr_scheduler == False:
            missing, unexpected = load_checkpoint(args.resume_from, model)
        else:
            start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                               model=model,
                                                               optimizer=optimizer,
                                                               lr_scheduler=lr_scheduler,
                                                               )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model,)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()
