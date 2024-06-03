import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms


class MLP_Encoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_layers=2,
                 hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()

    def forward(self, x):
        # x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        # cov = self.cov_m(self.cov_out(output))
        cov = 0.1 + 0.9*self.cov_m(self.cov_out(output))
        return mean, cov


class MLP_Decoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_layers=2,
                 hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        # x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))
        # cov = torch.exp(self.cov_out(output))

        return mean, cov

class MLP_Z_Z_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()
        # self.cov_m = nn.ELU()

    def forward(self, x):
        # x = torch.swapaxes(x, -2, -1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output))
        # mean = torch.swapaxes(mean, -2, -1)
        # cov = torch.swapaxes(cov, -2, -1)

        return mean, cov

class MultiFidelityModel(nn.Module):
    def __init__(self, levels, input_dim, output_dims, **model_kwargs):
        super().__init__()
        self.device = torch.device(model_kwargs.get('device', 'cpu'))
        # self.device = torch.device("cuda")
        self.to(self.device)

        self.input_dim = input_dim
        self.hidden_dim = int(model_kwargs.get('hidden_dim', 32))
        self.z_dim = int(model_kwargs.get('z_dim', 32))
        self.hidden_layers = int(model_kwargs.get('hidden_layers', 2))

        self.context_percentage_low = float(
            model_kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(
            model_kwargs.get('context_percentage_high', 0.5))

        for level in range(1, levels+1):
            output_dim = output_dims[level-1]
            setattr(self, f"l{level}_output_dim", output_dim)
            setattr(self, f"l{level}_encoder_model", MLP_Encoder(
                self.input_dim + output_dim, self.z_dim, self.hidden_layers, self.hidden_dim).to(self.device))
            setattr(self, f"l{level}_decoder_model", MLP_Decoder(
                self.z_dim + self.input_dim, output_dim, self.hidden_layers, self.hidden_dim).to(self.device))
            setattr(self, f"l{level}_z_z_encoder_model", MLP_Z_Z_Encoder(
                self.z_dim, self.z_dim, self.hidden_layers, self.hidden_dim).to(self.device))

        self.fid_lats = model_kwargs.get('fid_lats', None)
        if self.fid_lats: # reshape to latitude x longitude
            lat = int(self.fid_lats[-1])
            lon = int(int(output_dims[levels-1]) / lat)
            resizer = transforms.Resize((lat, lon), antialias=True)
        else: # reshape to a square
            n = int(getattr(self, f"l{levels}_output_dim") ** 0.5)
            resizer = transforms.Resize((n, n), antialias=True)
        self.resizer = resizer

    def split_context_target(self, x, y, context_percentage_low, context_percentage_high):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low, context_percentage_high)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask_c = np.random.choice(ind, size=n_context, replace=False)
        mask_t = np.delete(ind, mask_c)

        return x[mask_c], y[mask_c], x[mask_t], y[mask_t], mask_c, mask_t

    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(
            n, var.size(0)).normal_()).to(self.device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def z_to_y(self, x, zs, level):
        output = (getattr(self, f"l{level}_decoder_model"))(
            torch.cat([x, zs], dim=-1))
        return output

    def xy_to_r(self, x, y, level):
        r_mu, r_cov = (getattr(self, f"l{level}_encoder_model"))(
            torch.cat([x, y], dim=-1))
        return r_mu, r_cov

    def ba_z_agg(self, r_mu, r_cov, z_mu=None, z_cov=None):
        if z_mu is None:
            z_mu = torch.zeros(r_mu[0].shape).to(self.device)
            z_cov = torch.ones(r_cov[0].shape).to(self.device)
        v = r_mu - z_mu
        w_cov_inv = 1 / r_cov
        z_cov_new = 1 / (1 / z_cov + torch.sum(w_cov_inv, dim=0))
        z_mu_new = z_mu + z_cov_new * torch.sum(w_cov_inv * v, dim=0)
        return z_mu_new, z_cov_new

    def evaluate(self, xs, ys, z_mus, z_covs):
        mus = []
        covs = []
        levels = len(xs)

        for lvl in range(levels):
            zs = self.sample_z(z_mus[lvl], z_covs[lvl], xs[lvl].size(0))
            mu, cov = self.z_to_y(xs[lvl], zs, level=lvl+1)
            mus.append(mu)
            covs.append(cov)

        # Get model prediction with level-1 fidelity prediction
        res_pred_mu = self.reshape_outputs(mus[-2], level=levels)
        mean_output_mu = torch.mean(res_pred_mu, axis=0)
        model_pred = mean_output_mu + mus[-1] # residual + highest fidelity prediction

        result = {
            "targets": ys,
            "output_mus": mus,
            "output_covs": covs,
            "model_pred": model_pred
        }
        return result

    def forward(self, xs, ys, z_mu=None, z_cov=None):
        """
        xs: [l1_x_all, l2_x_all...] list of tensors
        ys: [l1_y_all, l2_y_all...] list of tensors
        l1_z_mu: tensors first level 
        l1_z_cov: tensors first level
        """
        results = {
            "targets": [],
            "output_mus": [],
            "output_covs": [],
            "z_mu_all": [],
            "z_cov_all": [],
            "z_mu_cs": [],
            "z_cov_cs": []
        }
        levels = len(xs)

        # initalize with first level
        z_mu_0 = z_mu[0]
        z_cov_0 = z_cov[0]

        for level in range(1, levels + 1):
            x = xs[level - 1]
            y = ys[level - 1]
            x_c, y_c, x_t, y_t, mask_c, mask_t = self.split_context_target(x, y, self.context_percentage_low, self.context_percentage_high)

            if level == levels:
                res_pred_mu = self.reshape_outputs(results["output_mus"][-1], level=levels)
                mean_output_mu = torch.mean(res_pred_mu, axis=0)
                y = y - mean_output_mu
                y_c = y[mask_c]
                y_t = y[mask_t]

            r_mu_all, r_cov_all = self.xy_to_r(x, y, level)
            r_mu_c, r_cov_c = self.xy_to_r(x_c, y_c, level)
            z_mu_all, z_cov_all = self.ba_z_agg(r_mu_all, r_cov_all, z_mu_0, z_cov_0)
            z_mu_c, z_cov_c = self.ba_z_agg(r_mu_c, r_cov_c, z_mu_all, z_cov_all)

            # Redefine z_mu_0, z_cov_0 for fidelities up to levels-1
            if level != levels:
                z_mu_0, z_cov_0 = getattr(self, f"l{level}_z_z_encoder_model")(z_mu_all)

            zs = self.sample_z(z_mu_all, z_cov_all, x_t.size(0))
            output_mu, output_cov = self.z_to_y(x_t, zs, level)

            results["targets"].append(y_t.clone())
            results["output_mus"].append(output_mu.clone())
            results["output_covs"].append(output_cov.clone())
            results["z_mu_all"].append(z_mu_all.clone())
            results["z_cov_all"].append(z_cov_all.clone())
            results["z_mu_cs"].append(z_mu_c.clone())
            results["z_cov_cs"].append(z_cov_c.clone())

        return results

    def reshape_outputs(self, output, level):
        B = output.shape[0]

        if self.fid_lats:
            output_reshaped = output.reshape(B, self.fid_lats[level-1], -1)
            pred_mu_resized = self.resizer(output_reshaped).flatten(start_dim=1)
        else:
            output_reshaped = output.reshape(B, int(output.shape[1]**0.5), -1)
            pred_mu_resized = self.resizer(output_reshaped).flatten(start_dim=1)
        
        return pred_mu_resized
