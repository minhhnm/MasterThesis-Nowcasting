import torch
import torch.nn as nn
from ...base import NowcastingLightningModule
import numpy as np
from .utils import extract_into_tensor
from .ema import EMA


def _check_finite(name, x):
    if not torch.is_tensor(x):
        raise RuntimeError(f"Provided {name} is not a tensor")

    if torch.isfinite(x).all():
        return
    # x is tensor and has infinite values
    finite_mask = torch.isfinite(x)
    infinite_vals = x[~finite_mask]

    if infinite_vals.numel() > 0:
        msg = (
            f"{name} has non-finite values | "
            f"shape={tuple(x.shape)} "
            f"number of non-finite values: {infinite_vals.shape}"
        )
    else:
        msg = f"{name} has no finite values at all | shape={tuple(x.shape)}"   # Should never happen

    raise RuntimeError(msg)


class LatentDiffusionNet(nn.Module):
    def __init__(self, conditioner, denoiser, parametrization = "eps"):
        super().__init__()
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.parametrization = parametrization

    def forward(self, x):
        # during training, noisy should be x_t
        # during inference, noisy should be noise
        t, noisy, latent_inputs = x
        _check_finite("forward/noisy", noisy)
        _check_finite("forward/latent_inputs", latent_inputs)

        condition = self.conditioner(latent_inputs)
        # _check_finite("forward/condition", condition)

        # if parametrization is eps, out is the predicted noise
        # if parametrization is x0, out is the guessed x0
        # if parametrization is v, out is the guessed v
        out = self.denoiser(noisy, t, context = condition)
        _check_finite("forward/out", out)
        return out


class LatentDiffusion(NowcastingLightningModule):
    def __init__(self, net, loss, scheduler, autoencoder, ema_config = {'use': True}, **kwargs):
        super().__init__(net, loss, **kwargs)
        self.save_hyperparameters(ignore=['net', 'loss', 'autoencoder'])
        self.scheduler = scheduler
        self.autoencoder = autoencoder # the LatentDiffusion class needs the autoencoder so that, in a parallel training setup, Lightning creates one instance of the autoencoder on each GPU

        # Freeze autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # register the schedules (i.e. the values of alpha, beta etc)
        self.register_schedule()

        if ema_config['use']:
            self.ema = EMA(self.net.denoiser, **ema_config['kwargs'])
            
    def register_schedule(self):
        
        schedule = self.scheduler.schedule(torch.float32, next(self.net.parameters()).device)
        
        # check if the ldm has already some saved buffers
        saved_buffers = dict(self.net.named_buffers())
        already_saved_and_different = [name for name in schedule.keys() 
                                       if (name in saved_buffers.keys() and (schedule[name] != saved_buffers[name]).any())
                                      ]
        if len(already_saved_and_different) > 0:
            raise AttributeError(f'The denoiser has already some different values for {already_saved_and_different}')
        
        for k in schedule.keys():
            self.net.denoiser.register_buffer(k, schedule[k])
    
    def training_logic(self, batch, batch_idx):
        
        inputs, true = batch
        _check_finite("training/inputs", inputs)
        _check_finite("training/true", true)

        with torch.no_grad():
            latent_inputs = self.autoencoder.encode(inputs)
            latent_true = self.autoencoder.encode(true)

        _check_finite("training/latent_inputs", latent_inputs)
        _check_finite("training/latent_true", latent_true)

        x0 = latent_true
        
        t, noise, x_t = self.q_sample(x0)
        _check_finite("training/noise", noise)
        _check_finite("training/x_t", x_t)
        
        if self.net.parametrization == 'eps':
            target = noise
        if self.net.parametrization == 'x0':
            target = x0

        _check_finite("training/target", target)
        
        model_output = self.net((t, x_t, latent_inputs))
        _check_finite("training/model_output", model_output)

        loss = self.loss(model_output, target)
        _check_finite("training/loss", loss)

        return loss

    def q_sample(self, x0, noise = None, t = None):
        '''generate noise target for training'''
        _check_finite("q_sample/x0", x0)

        if noise is None:
            noise = torch.randn_like(x0)
        if t is None:
            t = torch.randint(0, self.scheduler.timesteps, (x0.shape[0],), device=x0.device).long()

        _check_finite("q_sample/noise", noise)
        
        x_noisy = extract_into_tensor(self.net.denoiser.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract_into_tensor(self.net.denoiser.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

        _check_finite("q_sample/x_noisy", x_noisy)
        
        return t, noise, x_noisy

    def on_fit_start(self):
        if hasattr(self, 'ema'):
            self.ema.register()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, 'ema'):
            self.ema.update()

    def on_validation_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_validation_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()

    def on_test_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_test_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()
    
    def on_predict_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_predict_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()

    @classmethod
    def from_config(cls, config, autoencoder):

        from .scheduler import Scheduler
        from .unet import UNetModel
        from ..context.context import AFNONowcastNetCascade
        
        conditioner = AFNONowcastNetCascade(**config['conditioner'])
        denoiser = UNetModel(**config['denoiser'])
        net = LatentDiffusionNet(conditioner, denoiser)
        loss = nn.MSELoss()
        scheduler = Scheduler(**config['scheduler'])
        
        return cls(net, loss, scheduler, autoencoder,
                   ema_config = config['ema'],
                   optimizer_class = config['optimizer_class'],
                   optimizer_kwargs = config['optimizer_kwargs'],
                   lr_scheduler_config = config['lr_scheduler']
                  )