import torch
import numpy as np
import pytorch_lightning as pl
from dualss.modules.ema import LitEma
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from dualss.modules.diffusionmodules.model import Encoder, Decoder, Decoder_Attention
from dualss.modules.distributions.distributions import DiagonalGaussianDistribution

from dualss.util import instantiate_from_config

from packaging import version
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR 

laplacian_kernel = torch.tensor([[[[1/6.0,  4/6.0,  1/6.0],
                                    [4/6.0, -20/6.0, 4/6.0],
                                    [1/6.0,  4/6.0,  1/6.0]]]]).to('cuda') # !!!!!!!!!
def laplacian(x):
    return F.conv2d(x, laplacian_kernel, padding=1, groups=1)

class UV_RGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(2, 3, 3, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, uv):
        out = self.conv1(uv)
        out = self.relu(out)
        out = self.conv2(out)
        identity = self.conv3(uv)
        out += identity
        return self.relu(out)


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)

        ddconfig_texture = ddconfig.copy()
        ddconfig_texture["z_channels"] = ddconfig_texture["z_channels"] - 4
        self.decoder = Decoder_Attention(**ddconfig_texture)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim - 4, ddconfig_texture["z_channels"], 1)

        ddconfig_mask = ddconfig.copy()
        ddconfig_mask["z_channels"] = 4
        ddconfig_mask["out_ch"] = 1
        self.decoder_mask = Decoder(**ddconfig_mask)
        self.post_quant_conv_mask = torch.nn.Conv2d(4, ddconfig_mask["z_channels"], 1)


        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        
        self.uv_rgb = UV_RGB()
        self.D_u = torch.nn.Parameter(torch.tensor(0.15))
        self.D_v = torch.nn.Parameter(torch.tensor(0.07))
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        # mask
        quant_mask = self.post_quant_conv_mask(quant[:, :4])
        g_mask = self.decoder_mask(quant_mask)
        g_mask = (g_mask > 0.5).float()
        # texture
        quant = self.post_quant_conv(quant[:, 4:])
        rgb_uvfk = self.decoder(quant)
        rgb = rgb_uvfk[:, :3] + self.uv_rgb(rgb_uvfk[:, 3:5])
        
        return rgb * g_mask - 1.0 + g_mask 
    
    def decode_uvfk(self, quant): 
        quant = self.post_quant_conv(quant[:, 4:])
        rgb_uvfk = self.decoder(quant)
        return rgb_uvfk

    def decode_mask_only(self, quant, expand=True):
        quant_mask = self.post_quant_conv_mask(quant[:, :4])
        g_mask = self.decoder_mask(quant_mask)
        g_mask = (g_mask > 0.5).float() 
        if expand:
            kernel = torch.tensor( [[0., 1., 0.],
                                    [1., 1., 1.],
                                    [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3).to(g_mask.device)
            g_mask = F.conv2d(F.conv2d(1-g_mask, kernel, padding=1), kernel, padding=1)
            g_mask = 1 - (g_mask > 0).float()

        return g_mask

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant) # [b, 3, H, W]
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff    


    def decode_mask(self, quant): 
        # mask
        quant_mask = self.post_quant_conv_mask(quant[:, :4])
        g_mask = self.decoder_mask(quant_mask)
        
        # texture
        quant = self.post_quant_conv(quant[:, 4:])
        rgb_uvfk = self.decoder(quant)
        rgb = rgb_uvfk[:, :3] + self.uv_rgb(rgb_uvfk[:, 3:5])

        # PED
        u, v, F, k = rgb_uvfk[:, 3:4], rgb_uvfk[:, 4:5], rgb_uvfk[:, 5:6], rgb_uvfk[:, 6:7]
        # Laplacian
        lap_u = self.D_u * laplacian(u)
        lap_v = self.D_v * laplacian(v)
        # Gray-Scott PDE
        reaction = u * v * v
        ped_square = (lap_u - reaction + F * (1 - u)) ** 2 + (lap_v + reaction - (F + k) * v) ** 2

        return rgb, g_mask, ped_square
    
    def forward_for_loss(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec, g_mask, ped_square = self.decode_mask(quant) # [b, 4, H, W]
        if return_pred_indices:
            return dec, g_mask, ped_square, diff, ind
        return dec, g_mask, ped_square, diff    


    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        x = x.to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, g_mask, ped_square, qloss, ind = self.forward_for_loss(x, return_pred_indices=True) # 我修改，这里xrec有3+1通道

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, g_mask, ped_square, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, g_mask, ped_square, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, g_mask, ped_square, qloss, ind = self.forward_for_loss(x, return_pred_indices=True) # 我修改，这里xrec有3+1通道
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, g_mask, ped_square, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, g_mask, ped_square, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder_mask.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_mask.parameters())+
                                  list(self.uv_rgb.parameters())+ 
                                  [self.D_u, self.D_v], 
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            
            if self.scheduler_config["target"] == 'torch.optim.lr_scheduler.CosineAnnealingLR':
                T_max = self.scheduler_config["params"]["T_max"]
                eta_min = self.scheduler_config["params"]["eta_min"]
                scheduler = [
                    {
                        'scheduler': CosineAnnealingLR(opt_ae, T_max, eta_min),
                        'interval': 'step',
                        'frequency': 1
                    },
                    {
                        'scheduler': CosineAnnealingLR(opt_disc, T_max, eta_min),
                        'interval': 'step',
                        'frequency': 1
                    },
                ]

            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x) 
        h = self.quant_conv(h) 
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h


        quant_mask = self.post_quant_conv_mask(quant[:, :4])
        g_mask = self.decoder_mask(quant_mask) 
        g_mask = (g_mask > 0.5).float()

        # texture
        quant = self.post_quant_conv(quant[:, 4:])
        rgb_uvfk = self.decoder(quant)
        rgb = rgb_uvfk[:, :3] + self.uv_rgb(rgb_uvfk[:, 3:5])



        return rgb * g_mask - 1.0 + g_mask 


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x) # h.Size([16, 8, 32, 32])
        moments = self.quant_conv(h) # moments.Size([16, 8, 32, 32])， quant_conv=Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
        posterior = DiagonalGaussianDistribution(moments) 
        return posterior # posterior.Size([16, 4, 32, 32])

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input) 
        if sample_posterior:
            z = posterior.sample() # posterior.Size([16, 4, 32, 32])
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format)

        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
