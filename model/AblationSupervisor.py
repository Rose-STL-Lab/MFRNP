import numpy as np
from glob import glob
import torch
from model import AblationModel as AblationModel
from model import loss
from lib import utils
import yaml
from collections import defaultdict
import os
import wandb
import random
import datetime
import yaml
import logging
from ray import tune, train
import matplotlib.pyplot as plt


class Supervisor:
    def __init__(self, save_dir, ray_config=None, **kwargs):
        self.random_seed = kwargs["data"].get("random_seed", 42)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        self.levels = kwargs["data"].get("levels", 2)

        self.config = kwargs
        self.ray_config = ray_config
        if ray_config:
            self.config.update(ray_config)

        self.lr = kwargs["train"].get("base_lr", 0.001)
        self.curr_epoch = kwargs["train"].get("curr_epoch", 0)
        self.epochs = kwargs["train"].get("epochs", 2000)
        self.epsilon = kwargs["train"].get("epsilon", 1.0e-3)
        self.lr_decay_ratio = kwargs["train"].get("lr_decay_ratio", 0.1)
        self.max_grad_norm = kwargs["train"].get("max_grad_norm", 1)
        self.steps = kwargs["train"].get("steps", [100000])
        self.curr_patience = kwargs["train"].get("curr_patience", 0)
        self.patience = kwargs["train"].get("patience", 1000)
        self.fidelity_weight = kwargs["train"].get("fidelity_weight", 5)

        # data
        self.transform = kwargs["data"]["standard_transform"]
        input_dim, self.output_dims, self.data = utils.get_dataset(
            kwargs["data"]["data_path"], self.levels, transform=self.transform)

        # model
        self.model = AblationModel.MultiFidelityModel(
            self.levels, input_dim, self.output_dims, **self.config["model"])
        self.device = self.model.device
        logging.info("Device: " + str(self.device))

        # data loader
        self.train_loader = utils.MultiFidelityDataLoader(
            self.data, self.device, kwargs["data"]["batch_size"])
        self.valid_loader = utils.MultiFidelityDataLoader(
            self.data, self.device, kwargs["data"]["batch_size"], valid=True)
        self.test_loader = utils.MultiFidelityDataLoader(
            self.data, self.device, kwargs["data"]["batch_size"], test=True)

        # training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps=self.epsilon)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.steps, gamma=self.lr_decay_ratio)
        self.best_loss = float("inf")
        self.best_epoch = 0

        # save results
        self.save_dir = save_dir
        os.makedirs(os.path.join(self.save_dir, "valid_pred"), exist_ok=True)

    def inference(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.z_mu = state_dict["z_mu_all"]
        self.z_cov = state_dict["z_cov_all"]
        self.best_loss_dict, output = self._run_epoch(
            self.test_loader, train=False)
        logging.info("Test: " + ", ".join(
            [f'{loss_name}: {values:.4f}' for loss_name, values in self.best_loss_dict.items()]))

    def save_model(self, best=False):
        # save model weightts, optimizer, and self.epoch
        self.config["curr_epoch"] = self.curr_epoch
        self.config["curr_patience"] = self.curr_patience
        self.config["best_loss"] = self.best_loss
        self.config["best_epoch"] = self.best_epoch

        if best:
            save_dir = os.path.join(self.save_dir, "best.pt")
            with open(os.path.join(self.save_dir, "best_config.yaml"), 'w') as file:
                yaml.dump(self.config, file)
        else:
            save_dir = os.path.join(self.save_dir, "checkpoint.pt")
            with open(os.path.join(self.save_dir, "checkpoint.yaml"), 'w') as file:
                yaml.dump(self.config, file)
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "z_mu_all": self.z_mu,
            "z_cov_all": self.z_cov
        }
        torch.save(state, save_dir)
        # logging.info(f"Model saved to {save_dir}")

    def train(self):
        # Initalize l1_z_mu and l1_z_cov
        # self.z_mu and self.z_cov will be rewritten to [l1_z, l2_z...] during training for inference
        self.z_mu = [torch.zeros(self.model.z_dim).to(self.device)]
        self.z_cov = [torch.ones(self.model.z_dim).to(self.device)]

        for epoch in range(self.curr_epoch, self.epochs):
            self.curr_epoch = epoch
            train_loss_dict, train_output = self._run_epoch(
                self.train_loader, train=True)
            valid_loss_dict, valid_output = self._run_epoch(
                self.valid_loader, train=False)

            if self.config["wandb_log"]:
                for loss_name, values in train_loss_dict.items():
                    wandb.log({f"train_{loss_name}": values}, commit=False)
                for i, (loss_name, values) in enumerate(valid_loss_dict.items()):
                    commit = i == len(valid_loss_dict) - 1
                    wandb.log({f"valid_{loss_name}": values}, commit=commit)

            if epoch % 100 == 0:
                logging.info("Epoch: " + str(epoch))
                logging.info("Train: " + ", ".join(
                    [f'{loss_name}: {values:.4f}' for loss_name, values in train_loss_dict.items()]))
                logging.info("Valid: " + ", ".join(
                    [f'{loss_name}: {values:.4f}' for loss_name, values in valid_loss_dict.items()]))

                # plot first example
                # self.plot_valid_output(valid_output, epoch)

            valid_loss = valid_loss_dict[f"l{self.levels}_nrmse_loss"].item()
            if self.ray_config:
                train.report({"loss": valid_loss})
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_dict, test_output = self._run_epoch(
                    self.test_loader, train=False)
                self.best_epoch = epoch
                self.curr_patience = 0
                self.save_model(best=True)
            else:
                self.curr_patience += 1
                if self.curr_patience == self.patience:
                    logging.info("Patience ran out...")
                    break

        logging.info("Epoch: " + str(epoch))
        logging.info(f"Best validation loss: {self.best_loss:.4f}")
        logging.info("Test: " + ", ".join(
            [f'{loss_name}: {values:.4f}' for loss_name, values in self.best_loss_dict.items()]))
        self.save_model()

    def _run_epoch(self, loader, train=False):
        """
        Generic method to run an epoch of training, validation, or testing.
        """
        if train:
            self.train_loader.shuffle()

        epoch_loss_dict = defaultdict(list)

        for xs, ys in loader:
            if train:
                self.model.train()
                self.optimizer.zero_grad()
                # udpate latent variables for next pass
                output = self.model(xs, ys, self.z_mu, self.z_cov)

                loss_value = self.calculate_loss(
                    output, epoch_loss_dict, train=True)
                loss_value.backward()
                self.optimizer.step()

                # update z_mu and z_cov every iteration
                self.z_mu = [x.detach() for x in output["z_mu_all"]]
                self.z_cov = [x.detach() for x in output["z_cov_all"]]
            else:
                self.model.eval()
                with torch.no_grad():
                    output = self.model.evaluate(
                        xs, ys, self.z_mu, self.z_cov)
                    loss_value = self.calculate_loss(
                        output, epoch_loss_dict, train=False)

        # Format epoch loss and average the loss_dict
        for loss_name, values in epoch_loss_dict.items():
            epoch_loss_dict[loss_name] = np.mean(values)

        return epoch_loss_dict, output

    def calculate_loss(self, output, epoch_loss_dict, train):
        """
        Calculate and update the loss for the given epoch, for either training or testing.
        """
        def compute_rmse(pred, target, scaler, level):
            if self.transform:
                target_scaled = scaler.inverse_transform(
                    target.cpu().detach().numpy())
                pred_scaled = scaler.inverse_transform(
                    pred.cpu().detach().numpy())
                rmse = loss.rmse_metric(pred_scaled, target_scaled)
                nrmse = rmse / scaler.std
                epoch_loss_dict[f"l{level}_nrmse_loss"].append(np.mean(nrmse))
            else:
                rmse = loss.rmse_metric(
                    pred.cpu().detach().numpy(), target.cpu().detach().numpy())
            epoch_loss_dict[f"l{level}_rmse_loss"].append(np.mean(rmse))

        total_loss_value = 0

        for level in range(1, self.levels+1):
            scaler = self.data["scaler_y"][level-1]
            if train:
                nll_loss = loss.nll_loss(output["output_mus"][level-1], output["output_covs"]
                                         [level-1], output["targets"][level-1], return_numpy=False)
                kld_loss = loss.kld_gaussian_loss(
                    output["z_mu_all"][level-1], output["z_cov_all"][level-1], output["z_mu_cs"][level-1], output["z_cov_cs"][level-1])
                compute_rmse(output["output_mus"][level-1],
                             output["targets"][level-1], scaler, level)

                epoch_loss_dict[f"l{level}_nll_loss"].append(nll_loss.item())
                epoch_loss_dict[f"l{level}_kld_loss"].append(kld_loss.item())

                if level == self.levels:
                    # if self.curr_epoch >= 5000:
                    loss_value = nll_loss * self.fidelity_weight + kld_loss
                    # else:
                    #     loss_value = 0
                else:
                    loss_value = nll_loss + kld_loss
            else:
                if level == self.levels:
                    pred = output["model_pred"]
                else:
                    pred = output["output_mus"][level-1]
                nll_loss = loss.nll_loss(
                    pred, output["output_covs"][level-1], output["targets"][level-1], return_numpy=True)
                compute_rmse(pred, output["targets"][level-1], scaler, level)
                epoch_loss_dict[f"l{level}_nll_loss"].append(nll_loss.item())
                loss_value = nll_loss

            total_loss_value += loss_value

        return total_loss_value

    # def plot_valid_output(self, output, epoch):

    #     plot_np = output["model_pred"][0].cpu().detach().numpy()
    #     n = int(plot_np.shape[0]**0.5)
    #     plot_np = plot_np.reshape(n, n)
    #     plt.clf()

    #     plt.imshow(plot_np)
    #     plt.colorbar()
    #     plt.savefig(os.path.join(self.save_dir,
    #                 f"valid_pred/epoch_{epoch}.png"))
