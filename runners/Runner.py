import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.XTRA import XTRA


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = XTRA(args)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda:0")
            self.model = self.model.to(self.device)

    # Trong Runner.py, phương thức make_optimizer():
    def make_optimizer(self):
        # Get parameter IDs to check for identity rather than trying tensor comparison
        beta_param_ids = [id(p) for p in [self.model.beta_en, self.model.beta_cn]]
        
        # Separate beta parameters and other parameters using parameter IDs
        beta_params = [p for p in self.model.parameters() if id(p) in beta_param_ids]
        other_params = [p for p in self.model.parameters() if id(p) not in beta_param_ids]
        
        args_dict = {
            'params': [
                {'params': other_params, 'lr': self.args.learning_rate},
                {'params': beta_params, 'lr': self.args.learning_rate }  # Lower learning rate for beta
            ]
        }
        
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)
        else:
            raise NotImplementedError(self.args.lr_scheduler)

        return lr_scheduler

    def train(self, data_loader):

        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()

        # Add these parameters
        max_grad_norm = 1.0  # Adjust this value as needed
 
        if 'lr_scheduler' in self.args:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):

            sum_loss = 0.

            loss_rst_dict = defaultdict(float)

            self.model.train()
            for batch_data in data_loader:
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                cluster_info = {
                'cluster_en': batch_data['cluster_en'],
                'cluster_cn': batch_data['cluster_cn']
                }
                document_info = {
                'doc_embedding_en': batch_data['doc_embedding_en'],
                'doc_embedding_cn': batch_data['doc_embedding_cn']
                }
            
                # Trong Runner.py, train method:
                rst_dict = self.model(batch_bow_en, batch_bow_cn, document_info, cluster_info)
                batch_loss = rst_dict['loss']

                for key in rst_dict:
                    if 'loss' in key:
                        loss_rst_dict[key] += rst_dict[key]

                optimizer.zero_grad()
                batch_loss.backward()
                # Add gradient clipping before optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_en)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            sum_loss /= data_size

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / num_batch :.3f}'

            print(output_log)

        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()
        return beta_en, beta_cn


    def get_theta(self, bow, lang):
        """Get topic distribution from BOW"""
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.args.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                # Đảm bảo batch_bow ở đúng device
                if isinstance(batch_bow, np.ndarray):
                    batch_bow = torch.tensor(batch_bow, dtype=torch.float)
                if hasattr(batch_bow, 'device') and batch_bow.device != self.device:
                    batch_bow = batch_bow.to(self.device)
                theta = self.model.get_theta(batch_bow, lang=lang)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)

    # Keep existing code and modify Runner.py:
    def test(self, dataset):
        theta_en = self.get_theta(dataset.bow_en, lang='en')
        theta_cn = self.get_theta(dataset.bow_cn, lang='cn')
        return theta_en, theta_cn