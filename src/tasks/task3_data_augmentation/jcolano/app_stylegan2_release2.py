
import math
from typing import Tuple, Optional, List, Iterator
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from pathlib import Path
import torchvision
from PIL import Image
from labml import tracker, lab, monit, experiment
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.stylegan import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.utils import cycle_dataloader


class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)
        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss


class GradientPenalty(nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):  
        super().__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):   
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):  
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        return self.net(z)


class StyleBlock(nn.Module):
    def __init__(self, d_latent: int, in_features: int, out_features: int):  
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]): 
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)

        x = self.pad(x)

        x = F.conv2d(x, self.kernel)

        return x.view(b, c, h, w)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))



class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim=1)


class ToRGB(nn.Module):
    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

class GeneratorBlock(nn.Module):
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()

        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):  
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class Generator(nn.Module):
    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):  
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new

        return rgb


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features): 
        super().__init__()
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample()

        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class Discriminator(nn.Module):
    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512): 
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, image_size: int): 
        super().__init__()

        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()

    discriminator: Discriminator
    generator: Generator
    mapping_network: MappingNetwork

    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss

    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam

    gradient_penalty = GradientPenalty()
    gradient_penalty_coefficient: float = 10.

    path_length_penalty: PathLengthPenalty

    loader: Iterator

    batch_size: int = 32
    d_latent: int = 512
    image_size: int = 128
    mapping_network_layers: int = 8
    learning_rate: float = 1e-3
    mapping_network_learning_rate: float = 1e-5
    gradient_accumulate_steps: int = 1
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    style_mixing_prob: float = 0.9

    training_steps: int = 150_000

    n_gen_blocks: int


    lazy_gradient_penalty_interval: int = 4
    
    lazy_path_penalty_interval: int = 32

    lazy_path_penalty_after: int = 5_000

    log_generated_interval: int = 50

    save_checkpoint_interval: int = 200


    mode: ModeState
    log_layer_outputs: bool = False

    dataset_path: str = str('./images2/')

    def init(self):
        dataset = Dataset(self.dataset_path, self.image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=0,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        self.loader = cycle_dataloader(dataloader)

        log_resolution = int(math.log2(self.image_size))

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        if self.log_layer_outputs:
            hook_model_outputs(self.mode, self.discriminator, 'discriminator')
            hook_model_outputs(self.mode, self.generator, 'generator')
            hook_model_outputs(self.mode, self.mapping_network, 'mapping_network')

        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )

        tracker.set_image("generated", True)

    def get_w(self, batch_size: int):
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_noise(self, batch_size: int):  
        noise = []
        resolution = 4

        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            noise.append((n1, n2))

            resolution *= 2

        return noise

    def generate_images(self, batch_size: int):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)

        images = self.generator(w, noise)

        return images, w

    def step(self, idx: int):
        with monit.section('Discriminator'):
            self.discriminator_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                with self.mode.update(is_log_activations=(idx + 1) % self.log_generated_interval == 0):
                    generated_images, _ = self.generate_images(self.batch_size)
                    fake_output = self.discriminator(generated_images.detach())

                    real_images = next(self.loader).to(self.device)
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()
                    real_output = self.discriminator(real_images)

                    real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                    disc_loss = real_loss + fake_loss

                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add('loss.gp', gp)
                        disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

                    disc_loss.backward()

                    tracker.add('loss.discriminator', disc_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('discriminator', self.discriminator)

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.discriminator_optimizer.step()

        with monit.section('Generator'):
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                generated_images, w = self.generate_images(self.batch_size)
                fake_output = self.discriminator(generated_images)

                gen_loss = self.generator_loss(fake_output)

                if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                    plp = self.path_length_penalty(w, generated_images)
                    if not torch.isnan(plp):
                        tracker.add('loss.plp', plp)
                        gen_loss = gen_loss + plp

                gen_loss.backward()

                tracker.add('loss.generator', gen_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('generator', self.generator)
                tracker.add('mapping_network', self.mapping_network)

            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()

        if (idx + 1) % self.log_generated_interval == 0:
            tracker.add('generated', torch.cat([generated_images[:6], real_images[:3]], dim=0))
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()

        tracker.save()

    def train(self):
        for i in monit.loop(self.training_steps):
            self.step(i)
            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()


def main():
    experiment.create(name='stylegan2')
    configs = Configs()

    experiment.configs(configs, {
        'device.cuda_device': 0,
        'image_size': 64,
        'log_generated_interval': 50
    })

    configs.init()
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with experiment.start():
        configs.train()


if __name__ == '__main__':
    main()
    