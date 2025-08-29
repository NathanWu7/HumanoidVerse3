from typing import Tuple, Dict, Type
import numpy as np
try:
    import torch
except ImportError:
    pass

class NoiseProcess:
    def __init__(self, shape: Tuple[int, ...], tensor_type: str = "np", **kwargs):
        self.shape = shape
        self.x = None
        if tensor_type == "np":
            self.randn = np.random.randn
            self.zeros = np.zeros
        elif tensor_type.startswith("torch_"):
            # syntax: torch_<device>, torch_cpu, torch_cuda
            self.randn = lambda *shape: torch.randn(*shape, device=tensor_type.split("_")[-1])
            self.zeros = lambda *shape: torch.zeros(*shape, device=tensor_type.split("_")[-1])
        else:
            raise ValueError(f"Invalid tensor type: {tensor_type}")
        
    def reset(self):
        raise NotImplementedError

    def reset_part(self, mask: np.ndarray):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def stationary_distribution(self):
        raise NotImplementedError
    
class EmptyNoise(NoiseProcess):
    def __init__(self, shape: Tuple[int, ...], **kwargs):
        super().__init__(shape, **kwargs)
        self.x = self.zeros(shape)

    def reset(self):
        pass
    
    def step(self):
        return self.x
    
class OUProcess(NoiseProcess):
    def __init__(self, shape: Tuple[int, ...], mu: float, sigma: float, theta: float, dt: float, **kwargs):
        super().__init__(shape, **kwargs)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    def __repr__(self):
        return f"OUProcess(shape={self.shape}, mu={self.mu}, sigma={self.sigma}, theta={self.theta}, dt={self.dt})"

    def stationary_distribution(self):
        return (self.mu, self.sigma / np.sqrt(2 * self.theta))

    def reset(self):
        self.x = self.randn(*self.shape) * self.sigma / np.sqrt(2 * self.theta) + self.mu
        
    def reset_part(self, mask: np.ndarray):
        # mask: bool array with same shape as self.x
        self.x[mask] = self.randn(torch.sum(mask)) * self.sigma / np.sqrt(2 * self.theta) + self.mu

    def step(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * self.randn(*self.shape) * np.sqrt(self.dt)
        return self.x

class WhiteNoise(NoiseProcess):
    def __init__(self, shape: Tuple[int, ...], mu: float = 0, sigma: float = 1, **kwargs):
        super().__init__(shape, **kwargs)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def __repr__(self):
        return f"WhiteNoise(shape={self.shape}, mu={self.mu}, sigma={self.sigma})"

    def stationary_distribution(self):
        return (self.mu, self.sigma)

    def reset(self):
        self.x = self.randn(*self.shape) * self.sigma + self.mu

    def step(self):
        self.x = self.randn(*self.shape) * self.sigma + self.mu
        return self.x
    
class PinkNoise(NoiseProcess):
    def __init__(self, shape: tuple, mu: float = 0, sigma: float = 1, ncols: int = 16, **kwargs):
        super().__init__(shape, **kwargs)
        self.nrows = int(np.prod(shape))
        self.ncols = ncols
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.randn(self.nrows, self.ncols)
        self.step()

    def step(self):
        col_indices = np.random.geometric(0.5, size=self.nrows)
        col_indices[col_indices >= self.ncols] = 0
        
        self.state[np.arange(self.nrows), col_indices] = self.randn(self.nrows)
        self.state[:,0] = self.randn(self.nrows)
        
        pink_noise = np.sum(self.state, axis=-1)/np.sqrt(self.ncols)
        self.x = (self.mu + self.sigma*pink_noise).reshape(self.shape)
        return self.x

noise_process_dict:Dict[str,Type[NoiseProcess]] = {
    "ou": OUProcess,
    "white": WhiteNoise,
    "pink": PinkNoise,
    "empty": EmptyNoise,
}


class RadialPerturbation:
    # Thanks to `https://github.com/minyoungkim21/vmf-lib`, we rewrite the implementation of vMF from it.
    def __init__(self, sigma:float, kappa:float, rsf:int=10):
        self.sigma = torch.tensor(sigma)
        self.kappa = torch.tensor(kappa)
        self.rsf = rsf
        
        self.mu_ln = torch.log(1/torch.sqrt(1+self.sigma**2))
        self.sigma_ln = torch.sqrt(torch.log(1+self.sigma**2))
        
    def _norm(self, input: torch.Tensor, p:int=2, dim:int=0, eps:float=1e-12):
        return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)
        
    def lognormal_sample(self, N:int):
        return torch.exp(torch.randn(N) * self.sigma_ln + self.mu_ln)
    
    def vMF_sample(self, vec_unit:torch.Tensor):
        # vec_unit: [N,d]: N unit vector
        N = vec_unit.shape[0]
        d = vec_unit.shape[1]
        kappa = self.kappa
        rsf = self.rsf
        
        v = torch.randn(N, d-1).to(vec_unit)
        v = v / self._norm(v, dim=1)
        
        kmr = np.sqrt( 4*kappa.item()**2 + (d-1)**2 )
        bb = (kmr - 2*kappa) / (d-1)
        aa = (kmr + 2*kappa + d - 1) / 4
        dd = (4*aa*bb)/(1+bb) - (d-1)*np.log(d-1)
        beta = torch.distributions.Beta( torch.tensor(0.5*(d-1)), torch.tensor(0.5*(d-1)) )
        uniform = torch.distributions.Uniform(0.0, 1.0)
        v0 = torch.tensor([]).to(vec_unit)
        while len(v0) < N:
            eps = beta.sample([1, rsf*(N-len(v0))]).squeeze().to(vec_unit)
            uns = uniform.sample([1, rsf*(N-len(v0))]).squeeze().to(vec_unit)
            w0 = (1 - (1+bb)*eps) / (1 - (1-bb)*eps)
            t0 = (2*aa*bb) / (1 - (1-bb)*eps)
            det = (d-1)*t0.log() - t0 + dd - uns.log()
            v0 = torch.cat([v0, torch.tensor(w0[det>=0]).to(vec_unit)])
            if len(v0) > N:
                v0 = v0[:N]
                break
        v0 = v0.reshape([N,1])
    
        samples = torch.cat([v0, (1-v0**2).sqrt()*v], 1)

        e1mu = torch.zeros(N, d).to(vec_unit)
        e1mu[:,0] = 1.0
        e1mu = e1mu - vec_unit
        e1mu = e1mu / self._norm(e1mu, dim=1)
        
        samples -= 2* (samples * e1mu).sum(dim=1, keepdim=True) * e1mu
        
        # samples = samples - 2 * torch.einsum('nd,nmd->nd', samples, e1mu @ e1mu.transpose(-2,-1))

        # samples = samples - 2 * (samples @ e1mu) @ e1mu.t()
        
        return samples
    
    def __call__(self, vec):
        N = vec.shape[0]
        vec_norm = self._norm(vec, dim=1) + 1e-10
        vec_unit = vec / vec_norm
        
        
        vMF_samples = self.vMF_sample(vec_unit)
        LN_samples = self.lognormal_sample(N).reshape([N,1])
        # print(f'{vMF_samples.shape=}')
        # print(f'{LN_samples.shape=}, {LN_samples}')
        # print(f'{vec_norm.shape=}')
        return vMF_samples * LN_samples * vec_norm