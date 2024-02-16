import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class ConvNetwork(nn.Module):
            def __init__(self, n_c=1, n_f=64, l=0.2, sn= True):
                super(ConvNetwork, self).__init__()
                if not sn :
                    self.f = nn.Sequential(
                        nn.Conv2d(n_c, n_f, 3, 1, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 8, 1, 4, 1, 0))
                else :
                    self.f = nn.Sequential(
                        spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 8, 1, 4, 1, 0, bias=False)))



            def forward(self, x):
                return self.f(x).squeeze()
