import torch
import torch.nn as nn

from .blocks import DoubleConvCat, DownCat, Up, OutConv


class CVAE(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(CVAE, self).__init__()
        in_ch = cfg.OUTPUT.CHANNELS
        out_ch = in_ch
        self.e_ch = cfg.MODEL.CVAE.ENCODER.OUT_CHANNELS
        self.d_ch = cfg.MODEL.CVAE.DECODER.OUT_CHANNELS
        self.latent_dim = cfg.MODEL.CVAE.LATENT.DIMENSIONS
        self.latent_input_dim = cfg.MODEL.CVAE.LATENT.INPUT_DIM

        # Encoder
        self.down1 = DownCat(in_ch, self.e_ch[0])         
        self.down2 = DownCat(self.e_ch[0], self.e_ch[1]) 
        self.down3 = DownCat(self.e_ch[1], self.e_ch[2]) 
        self.down4 = DownCat(self.e_ch[2], self.e_ch[3])  
        self.down5 = DownCat(self.e_ch[3], self.e_ch[4]) 

        # Fully connected layers for code mean and log variance
        self.mu = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)
        self.logvar = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)

        # Fully connected layer from code to decoder
        self.decoder_in = nn.Linear(self.latent_dim, self.d_ch[0]*self.latent_input_dim[0]*self.latent_input_dim[1])

        # Decoder
        self.d_inc = DoubleConvCat(self.d_ch[0], self.d_ch[0], linear=True)
        self.up1 = Up(self.d_ch[0], self.d_ch[1], bilinear, linear=True)   
        self.up2 = Up(self.d_ch[1], self.d_ch[2], bilinear, linear=True)   
        self.up3 = Up(self.d_ch[2], self.d_ch[3], bilinear, linear=True)   
        self.up4 = Up(self.d_ch[3], self.d_ch[4], bilinear, linear=True)   

        self.out = OutConv(self.d_ch[4], out_ch)


    def encode(self, feature_maps, x):
        x1, x2, x3, x4, x5, _ = feature_maps
        
        x = self.down1(x, x5)
        x = self.down2(x, x4)
        x = self.down3(x, x3)
        x = self.down4(x, x2)
        x = self.down5(x, x1)

        return x

    def decode(self, feature_maps, z):
        x1, x2, x3, x4, x5, _ = feature_maps

        x_flattened = self.decoder_in(z)
        x = x_flattened.view(-1, self.d_ch[0], self.latent_input_dim[0], self.latent_input_dim[1])
        x = self.d_inc(x, x1)
        x = self.up1(x, x2)
        x = self.up2(x, x3)
        x = self.up3(x, x4)
        x = self.up4(x, x5)
        out = self.out(x)

        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std)

        return mu + std * eps
    
    def sample(self, zero_code=True):
        if zero_code:
            z = torch.zeros((1, self.latent_dim))
        else:
            z = torch.randn((1, self.latent_dim))
        return z.cuda() if torch.cuda.is_available() else z

    def forward(self, feature_maps, x=None):
        if x is None: # Testing (no ground truth depth maps)
            z = self.sample(zero_code=True)
            out = self.decode(feature_maps, z)
            mu = logvar = None
        else: # Training
            x = self.encode(feature_maps, x)

            x_flattened = torch.flatten(x, start_dim=1)
            
            mu = self.mu(x_flattened)
            logvar = self.logvar(x_flattened)
            z = self.reparameterize(mu, logvar)

            out = self.decode(feature_maps, z)

        return (out, mu, logvar)


class CVAEResNet18(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(CVAEResNet18, self).__init__()
        in_ch = cfg.OUTPUT.CHANNELS
        out_ch = in_ch
        self.e_ch = cfg.MODEL.CVAE.ENCODER.OUT_CHANNELS
        self.d_ch = cfg.MODEL.CVAE.DECODER.OUT_CHANNELS
        self.latent_dim = cfg.MODEL.CVAE.LATENT.DIMENSIONS
        self.latent_input_dim = cfg.MODEL.CVAE.LATENT.INPUT_DIM

        # Encoder
        self.down1 = DownCat(in_ch, self.e_ch[0])         
        self.down2 = DownCat(self.e_ch[0], self.e_ch[1]) 
        self.down3 = DownCat(self.e_ch[1], self.e_ch[2]) 
        self.down4 = DownCat(self.e_ch[2], self.e_ch[3])  

        # Fully connected layers for code mean and log variance
        self.mu = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)
        self.logvar = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)
         
        # Fully connected layer from code to decoder
        self.decoder_in = nn.Linear(self.latent_dim, self.d_ch[0]*self.latent_input_dim[0]*self.latent_input_dim[1])

        # Decoder
        self.d_inc = DoubleConvCat(self.d_ch[0], self.d_ch[0])
        self.up1 = Up(self.d_ch[0], self.d_ch[1], bilinear)   
        self.up2 = Up(self.d_ch[1], self.d_ch[2], bilinear)   
        self.up3 = Up(self.d_ch[2], self.d_ch[3], bilinear)     

        self.out = nn.Sequential(
            OutConv(self.d_ch[3], out_ch),
            nn.Sigmoid()
        )

    def encode(self, feature_maps, x):
        x1, x2, x3, x4, _ = feature_maps
        
        x = self.down1(x, x4)
        x = self.down2(x, x3)
        x = self.down3(x, x2)
        x = self.down4(x, x1)

        return x

    def decode(self, feature_maps, z):
        x1, x2, x3, x4, _ = feature_maps

        x_flattened = self.decoder_in(z)
        x = x_flattened.view(-1, self.d_ch[0], self.latent_input_dim[0], self.latent_input_dim[1])
        x = self.d_inc(x, x1)
        x = self.up1(x, x2)
        x = self.up2(x, x3)
        x = self.up3(x, x4)
        out = self.out(x)

        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std)

        return mu + std * eps
    
    def sample(self, zero_code=True):
        if zero_code:
            z = torch.zeros((1, self.latent_dim))
        else:
            z = torch.randn((1, self.latent_dim))
        return z.cuda() if torch.cuda.is_available() else z

    def forward(self, feature_maps, x=None):
        if x is None: # Testing (no ground truth depth maps)
            z = self.sample(zero_code=True)
            out = self.decode(feature_maps, z)
            mu = logvar = None
        else: # Training
            x = self.encode(feature_maps, x)

            x_flattened = torch.flatten(x, start_dim=1)
            
            mu = self.mu(x_flattened)
            logvar = self.logvar(x_flattened)
            z = self.reparameterize(mu, logvar)

            out = self.decode(feature_maps, z)

        return (out, mu, logvar)