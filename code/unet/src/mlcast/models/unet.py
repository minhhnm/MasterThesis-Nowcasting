"""U-Net architecture for deterministic spatio-temporal nowcasting.

Provides a plain PyTorch U-Net model adapted to the mlcast
nowcasting interface. The model receives a sequence in
``(batch, time, channels, height, width)`` format, channel-stacks the input
frames, and predicts future frames autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped


class DoubleConv(nn.Module):
    """Two consecutive convolutional layers with ReLU activations.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int, optional
        Convolution kernel size. Default is ``3``.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels_out height width"]:
        """Forward pass."""
        return self.net(x)


class DownBlock(nn.Module):
    """Downsampling U-Net block.

    Applies max pooling followed by :class:`DoubleConv`.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, output_channels, kernel_size=kernel_size),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels_out height_out width_out"]:
        """Forward pass."""
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling U-Net block with skip connection."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(input_channels // 2 + skip_channels, output_channels, kernel_size=kernel_size)

    @staticmethod
    def _match_spatial_size(
        x: Float[torch.Tensor, "batch channels height width"],
        target: Float[torch.Tensor, "batch target_channels target_height target_width"],
    ) -> Float[torch.Tensor, "batch channels target_height target_width"]:
        """Pad or crop ``x`` so that its spatial size matches ``target``."""
        diff_h = target.shape[-2] - x.shape[-2]
        diff_w = target.shape[-1] - x.shape[-1]

        if diff_h > 0 or diff_w > 0:
            x = F.pad(
                x,
                [
                    max(diff_w // 2, 0),
                    max(diff_w - diff_w // 2, 0),
                    max(diff_h // 2, 0),
                    max(diff_h - diff_h // 2, 0),
                ],
            )

        if diff_h < 0:
            crop_top = (-diff_h) // 2
            x = x[..., crop_top : crop_top + target.shape[-2], :]

        if diff_w < 0:
            crop_left = (-diff_w) // 2
            x = x[..., :, crop_left : crop_left + target.shape[-1]]

        return x

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch channels height width"],
        skip: Float[torch.Tensor, "batch skip_channels skip_height skip_width"],
    ) -> Float[torch.Tensor, "batch channels_out skip_height skip_width"]:
        """Forward pass."""
        x = self.up(x)
        x = self._match_spatial_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetModel(nn.Module):
    """Autoregressive U-Net model adapted to the mlcast nowcasting API.

    The model uses a 2D U-Net internally. Past frames are stacked along the
    channel dimension, one future frame is predicted, and the input window is
    updated autoregressively until ``steps`` forecasts have been produced.

    Parameters
    ----------
    input_channels : int, optional
        Number of variables/channels per timestep. Default is ``1``.
    input_steps : int, optional
        Number of past timesteps provided as input. Default is ``6``.
    base_channels : int, optional
        Number of feature channels in the first U-Net level. Default is ``32``.
    num_blocks : int, optional
        Number of downsampling blocks. The spatial resolution is downsampled by
        a factor of ``2 ** num_blocks``. Default is ``4``.
    kernel_size : int, optional
        Convolution kernel size. Default is ``3``.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_steps: int = 6,
        base_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if input_channels < 1:
            raise ValueError("input_channels must be at least 1.")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.input_channels = input_channels
        self.input_steps = input_steps
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        stacked_channels = input_steps * input_channels
        channels = [base_channels * 2**i for i in range(num_blocks + 1)]

        self.input_conv = DoubleConv(stacked_channels, channels[0], kernel_size=kernel_size)
        self.down_blocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1], kernel_size=kernel_size) for i in range(num_blocks)]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i], channels[i], kernel_size=kernel_size)
                for i in reversed(range(num_blocks))
            ]
        )
        self.output_conv = nn.Conv2d(base_channels, input_channels, kernel_size=1)

    @staticmethod
    def _pad_to_divisor(
        x: Float[torch.Tensor, "batch channels height width"],
        divisor: int,
    ) -> tuple[Float[torch.Tensor, "batch channels padded_height padded_width"], int, int]:
        """Pad spatial dimensions so they are divisible by ``divisor``."""
        height, width = x.shape[-2:]
        pad_h = (divisor - height % divisor) % divisor
        pad_w = (divisor - width % divisor) % divisor

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        return x, pad_h, pad_w

    @jaxtyped(typechecker=beartype)
    def _predict_one_step(
        self,
        x: Float[torch.Tensor, "batch stacked_channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        """Predict one future timestep from the stacked input window."""
        height, width = x.shape[-2:]
        divisor = 2**self.num_blocks

        x, _pad_h, _pad_w = self._pad_to_divisor(x, divisor)

        skips = []
        x = self.input_conv(x)
        skips.append(x)

        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        skips_for_decoder = list(reversed(skips[:-1]))
        for block, skip in zip(self.up_blocks, skips_for_decoder, strict=True):
            x = block(x, skip)

        x = self.output_conv(x)
        return x[..., :height, :width]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
        steps: int,
        ensemble_size: int | None = None,
    ) -> Float[torch.Tensor, "batch forecast_steps channels height width"]:
        """Run an autoregressive forecast.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input tensor.
        steps : int
            Number of future timesteps to predict.
        ensemble_size : int or None, optional
            Accepted for compatibility with the mlcast model API. This U-Net is
            deterministic and ignores this argument.

        Returns
        -------
        preds : Float[torch.Tensor, "batch forecast_steps channels height width"]
            Forecast tensor.
        """
        del ensemble_size

        if steps < 1:
            raise ValueError("steps must be at least 1.")

        batch, input_steps, channels, height, width = x.shape

        if input_steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {input_steps}.")
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {channels}.")

        x_window = x.contiguous().reshape(batch, input_steps * channels, height, width)

        preds = []
        for _ in range(steps):
            pred = self._predict_one_step(x_window)
            preds.append(pred.unsqueeze(1))
            x_window = torch.cat([x_window[:, channels:], pred], dim=1)

        return torch.cat(preds, dim=1)


class StochasticUNetModel(nn.Module):
    """Noise-conditioned autoregressive U-Net for ensemble nowcasting.

    This model is similar to :class:`UNetModel`, but it appends random noise
    channels to the stacked input window before each one-step prediction. When
    ``ensemble_size > 1``, each ensemble member receives independent noise.

    For single-channel rainfall nowcasting, the probabilistic output shape is:

    ``(batch, forecast_steps, ensemble_size, height, width)``

    which matches the CRPS convention used by mlcast.

    Parameters
    ----------
    input_channels : int, optional
        Number of variables/channels per timestep. Default is ``1``.
    input_steps : int, optional
        Number of past timesteps provided as input. Default is ``6``.
    base_channels : int, optional
        Number of feature channels in the first U-Net level. Default is ``32``.
    num_blocks : int, optional
        Number of downsampling blocks. Default is ``4``.
    kernel_size : int, optional
        Convolution kernel size. Default is ``3``.
    noise_channels : int, optional
        Number of random noise channels appended at every autoregressive step.
        Default is ``4``.
    noise_scale : float, optional
        Standard deviation multiplier for the Gaussian noise. Default is ``1.0``.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_steps: int = 6,
        base_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
        noise_channels: int = 4,
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if input_channels < 1:
            raise ValueError("input_channels must be at least 1.")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if noise_channels < 1:
            raise ValueError("noise_channels must be at least 1.")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative.")

        self.input_channels = input_channels
        self.input_steps = input_steps
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.noise_channels = noise_channels
        self.noise_scale = noise_scale

        stacked_channels = input_steps * input_channels + noise_channels
        channels = [base_channels * 2**i for i in range(num_blocks + 1)]

        self.input_conv = DoubleConv(stacked_channels, channels[0], kernel_size=kernel_size)
        self.down_blocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1], kernel_size=kernel_size) for i in range(num_blocks)]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i], channels[i], kernel_size=kernel_size)
                for i in reversed(range(num_blocks))
            ]
        )
        self.output_conv = nn.Conv2d(base_channels, input_channels, kernel_size=1)

    @jaxtyped(typechecker=beartype)
    def _predict_one_step(
        self,
        x: Float[torch.Tensor, "batch stacked_channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        """Predict one future timestep from the stacked input window plus noise."""
        height, width = x.shape[-2:]
        divisor = 2**self.num_blocks

        x, _pad_h, _pad_w = UNetModel._pad_to_divisor(x, divisor)

        skips = []
        x = self.input_conv(x)
        skips.append(x)

        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        skips_for_decoder = list(reversed(skips[:-1]))
        for block, skip in zip(self.up_blocks, skips_for_decoder, strict=True):
            x = block(x, skip)

        x = self.output_conv(x)
        return x[..., :height, :width]

    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
        steps: int,
        ensemble_size: int | None = None,
    ) -> torch.Tensor:
        """Run an autoregressive stochastic forecast.

        If ``ensemble_size == 1``, returns:

        ``(batch, forecast_steps, channels, height, width)``

        If ``ensemble_size > 1`` and ``channels == 1``, returns:

        ``(batch, forecast_steps, ensemble_size, height, width)``

        The second form is intended for CRPS training.
        """
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        if ensemble_size is None:
            ensemble_size = 1

        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1.")

        batch, input_steps, channels, height, width = x.shape

        if input_steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {input_steps}.")
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {channels}.")

        if ensemble_size > 1 and channels != 1:
            raise ValueError(
                "StochasticUNetModel currently supports ensemble CRPS output only "
                "for one output channel. For multi-variable probabilistic forecasts, "
                "the loss/output convention should be extended explicitly."
            )

        x_window = x.contiguous().reshape(batch, input_steps * channels, height, width)

        if ensemble_size > 1:
            x_window = x_window.repeat_interleave(ensemble_size, dim=0)

        member_batch = x_window.shape[0]

        preds = []
        for _ in range(steps):
            noise = torch.randn(
                member_batch,
                self.noise_channels,
                height,
                width,
                dtype=x.dtype,
                device=x.device,
            )
            noise = noise * self.noise_scale

            model_input = torch.cat([x_window, noise], dim=1)
            pred = self._predict_one_step(model_input)

            preds.append(pred.unsqueeze(1))
            x_window = torch.cat([x_window[:, channels:], pred], dim=1)

        preds_tensor = torch.cat(preds, dim=1)

        if ensemble_size == 1:
            return preds_tensor

        preds_tensor = preds_tensor.reshape(
            batch,
            ensemble_size,
            steps,
            channels,
            height,
            width,
        )
        preds_tensor = preds_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()

        # For rainfall, channels == 1, so CRPS expects [B, T, M, H, W].
        return preds_tensor[:, :, :, 0]
