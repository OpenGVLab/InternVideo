import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.mu.device)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def log_prob(self, input):
        gaussian1 = torch.distributions.Normal(0, self.sigma1.to(input.device))
        gaussian2 = torch.distributions.Normal(0, self.sigma2.to(input.device))
        prob1 = torch.exp(gaussian1.log_prob(input))
        prob2 = torch.exp(gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
        
        
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, pi=0.5, sigma_1=None, sigma_2=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if sigma_1 is None or sigma_2 is None:
            sigma_1 = torch.FloatTensor([math.exp(-0)])
            sigma_2 = torch.FloatTensor([math.exp(-6)])
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.bias_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, act=torch.relu, pi=0.5, sigma_1=None, sigma_2=None):
        super(BayesianPredictor, self).__init__()
        self.output_dim = output_dim
        self.act = act
        self.bnn_layer = BayesianLinear(input_dim, output_dim, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def log_prior(self):
        return self.bnn_layer.log_prior

    def log_variational_posterior(self):
        return self.bnn_layer.log_variational_posterior

    def forward(self, input, npass=2, testing=False):
        npass_size = npass + 1 if testing else npass
        outputs = torch.zeros(npass_size, input.size(0), self.output_dim).to(input.device)
        log_priors = torch.zeros(npass_size).to(input.device)
        log_variational_posteriors = torch.zeros(npass_size).to(input.device)
        for i in range(npass):
            outputs[i] = self.bnn_layer(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        if testing:
            outputs[npass] = self.bnn_layer(input, sample=False)
        return outputs, log_priors, log_variational_posteriors


def get_uncertainty(outputs):
    # predict the aleatoric and epistemic uncertainties
    p = F.softmax(outputs, dim=-1) # N x B x C
    # compute aleatoric uncertainty
    p_diag = torch.diag_embed(p, offset=0, dim1=-2, dim2=-1) # N x B x C x C
    p_cov = torch.matmul(p.unsqueeze(-1), p.unsqueeze(-1).permute(0, 1, 3, 2))  # N x B x C x C
    uncertain_alea = torch.mean(p_diag - p_cov, dim=0)  # B x C x C
    # compute epistemic uncertainty 
    p_bar= torch.mean(p, dim=0)  # B x C
    p_diff_var = torch.matmul((p-p_bar).unsqueeze(-1), (p-p_bar).unsqueeze(-1).permute(0, 1, 3, 2))  # N x B x C x C
    uncertain_epis = torch.mean(p_diff_var, dim=0)  # B x C x C
    # transform the class-wise uncertainties into total uncertainty by matrix tracing
    uncertain_alea = torch.diagonal(uncertain_alea, dim1=-2, dim2=-1).sum(-1)
    uncertain_epis = torch.diagonal(uncertain_epis, dim1=-2, dim2=-1).sum(-1)
    return uncertain_alea, uncertain_epis