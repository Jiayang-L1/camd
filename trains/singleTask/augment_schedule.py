import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop

logger = logging.getLogger('CAMD')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class AugmentModel:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop().getMetics(args.dataset_name)
        self.recon_loss = MSE()

        self.alpha = args.alpha
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2

    def train(self, model, dataloader, current_time):
        params = []
        param_aug = []
        augment_name = []
        main_name = []
        for name, param in model.named_parameters():
            params.append(param)
            main_name.append(name)
            if 'augmenter' in name:
                param_aug.append(param)
                augment_name.append(name)
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        optimizer_aug = optim.Adam(param_aug, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        min_or_max = 'min'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for meta, text, image, labels in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer_aug.zero_grad()
                        optimizer.zero_grad()

                    left_epochs -= 1

                    text = torch.tensor(np.array(text), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    image = torch.tensor(np.array(image), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    meta = torch.tensor(np.array(meta), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    labels = torch.tensor(np.array(labels), dtype=torch.float32).to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(meta, text, image, self.beta_1)

                    loss_task = self.criterion(output['ori_logits'], labels)
                    loss_aug = self.criterion(output['aug_logits'], labels)
                    loss_vae = output['vae_loss']
                    loss_regularize = output['regularize_loss']

                    loss_similarity = self.cosine(output['invariant_text'], output['invariant_image'],
                                                  torch.tensor([1]).to(self.args.device))

                    loss_difference = self.cosine(output['specific_text'], output['invariant_text'],
                                                  torch.tensor([-1]).to(self.args.device)) + \
                                      self.cosine(output['specific_image'], output['invariant_image'],
                                                  torch.tensor([-1]).to(self.args.device)) + \
                                      self.cosine(output['specific_text'], output['specific_image'],
                                                  torch.tensor([-1]).to(self.args.device))

                    loss_recon = self.recon_loss(output['text'], output['recon_text']) + \
                                 self.recon_loss(output['image'], output['recon_image'])

                    loss = loss_task + loss_aug + self.beta_2 * (loss_vae + loss_regularize) + \
                           self.alpha * (loss_similarity + (1 / 3) * loss_difference + (1 / 2) * loss_recon)
                    loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model.parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += loss.item()

                    y_pred.append(output['ori_logits'].cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        optimizer_aug.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()
                    optimizer_aug.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)

            rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
            mae = torch.mean(torch.abs(pred - true)).item()

            train_results = [rmse, mae]
            train_results = [round(x, 4) for x in train_results]
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss ** 0.5, 4)} "
                f"RMSE:{train_results[0]} MAE:{train_results[1]}"
            )

            # validation
            val_results = self.test(model, dataloader['valid'])
            test_results = self.test(model, dataloader['test'])
            cur_valid = val_results[0]
            scheduler.step(val_results[0])

            # save each epoch model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = f'./pt/{current_time}-static.pth'
                torch.save(model.state_dict(), model_save_path)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epochs

    def test(self, model, dataloader):
        model.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for meta, text, image, labels in td:
                    text = torch.tensor(np.array(text), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    image = torch.tensor(np.array(image), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    meta = torch.tensor(np.array(meta), dtype=torch.float32).to(self.args.device).unsqueeze(1)
                    labels = torch.tensor(np.array(labels), dtype=torch.float32).to(self.args.device)

                    labels = labels.view(-1, 1)

                    output = model(meta, text, image, self.beta_1)

                    y_pred.append(output['ori_logits'].cpu())
                    y_true.append(labels.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
        mae = torch.mean(torch.abs(pred - true)).item()
        eval_results = [rmse, mae]
        eval_results = [round(x, 4) for x in eval_results]

        logger.info(
            f"RMSE:{eval_results[0]} MAE:{eval_results[1]}"
        )

        return eval_results
