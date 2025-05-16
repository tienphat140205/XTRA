import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.Encoder import Encoder

class XTRA(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_en = args.vocab_en
        self.vocab_cn = args.vocab_cn
        self.share_dim=args.share_dim
        
        # Lấy kích thước doc embedding
        doc_emb_dim = args.doc_embeddings_en[0].shape[0] if hasattr(args, 'doc_embeddings_en') and len(args.doc_embeddings_en) > 0 else 1024

        # Beta matrices
        self.beta_en = nn.Parameter(torch.tensor(args.beta_en).float(), requires_grad=True)
        self.beta_cn = nn.Parameter(torch.tensor(args.beta_cn).float(), requires_grad=True)
        
        
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)
        self.num_topic = args.num_topic
        self.temperature = args.temperature

        # Hyperparameters
        self.weight_infoncetheta = args.weight_infoncetheta 
        self.lambda_contrast = args.lambda_contrast
        self.infonce_alpha = args.infoncealpha
        # Prior parameters
        mu2_val = torch.tensor(args.mu_prior).float()
        var2_val = torch.tensor(args.var_prior).float()
        self.mu2 = nn.Parameter(mu2_val, requires_grad=False)
        self.var2 = nn.Parameter(var2_val, requires_grad=False)
        
        # Decoder BatchNorm layers
        self.decoder_bn_en = nn.BatchNorm1d(args.vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(args.vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False
  
        # Projection layers
        self.prj_beta_en = nn.Sequential(
            nn.Linear(self.vocab_size_en, 384),
            nn.Dropout(args.dropout),
        )
        self.prj_beta_cn = nn.Sequential(
            nn.Linear(self.vocab_size_cn, 384),
            nn.Dropout(args.dropout),
        )
        
        self.prj_rep = nn.Sequential(
            nn.Linear(args.num_topic, doc_emb_dim),
            nn.Dropout(args.dropout)
        )
        # Khai báo đúng
        self.mlp_en = nn.Sequential(
            nn.Linear(self.vocab_size_en, self.share_dim),
            nn.ReLU(),  # Thêm phi tuyến
            nn.Dropout(args.dropout)
        )
        self.mlp_cn = nn.Sequential(
            nn.Linear(self.vocab_size_cn, self.share_dim),
            nn.ReLU(),  # Thêm phi tuyến
            nn.Dropout(args.dropout)
        )
        self.prj_doc = nn.Sequential()
        # Sửa encoder để nhận input 1000 chiều
        self.encoder = Encoder(self.share_dim, args.num_topic, args.en1_units, args.dropout)
    # Sửa phương thức contrastive_loss - không đổi
    def contrastive_loss(self, theta_lang1, theta_lang2, cluster_info_lang1, cluster_info_lang2):
        """Calculate contrastive loss based on cluster information."""
        batch_size = theta_lang1.size(0)
        theta_all = torch.cat([theta_lang1, theta_lang2], dim=0)
        
        # Convert cluster info tensors to appropriate format
        if isinstance(cluster_info_lang1, torch.Tensor) and isinstance(cluster_info_lang2, torch.Tensor):
            cluster_all = torch.cat([cluster_info_lang1, cluster_info_lang2], dim=0)
        else:
            # Handle if cluster_info is a list of tensors or other formats
            cluster_all = []
            for c1, c2 in zip(cluster_info_lang1, cluster_info_lang2):
                if isinstance(c1, torch.Tensor) and isinstance(c2, torch.Tensor):
                    cluster_all.extend([c1, c2])
                else:
                    cluster_all.extend([torch.tensor(c1, device=theta_lang1.device), 
                                       torch.tensor(c2, device=theta_lang1.device)])
            cluster_all = torch.stack(cluster_all) if cluster_all else None
        
        if cluster_all is None:
            return torch.tensor(0.0, device=theta_lang1.device)
            
        theta_norm = F.normalize(theta_all, dim=-1)
        sim_matrix = torch.matmul(theta_norm, theta_norm.T) / self.temperature
        sim_exp = torch.exp(sim_matrix)
        
        eye_mask = torch.eye(2 * batch_size, device=theta_all.device, dtype=torch.bool)
        sim_exp = sim_exp * (~eye_mask).float()
        
        # Create positive mask based on cluster information
        pos_mask = (cluster_all.unsqueeze(0) == cluster_all.unsqueeze(1)).float()
        pos_mask = pos_mask * (~eye_mask).float()
        
        pos_sim = torch.sum(sim_exp * pos_mask, dim=1)
        neg_sum = torch.sum(sim_exp, dim=1) - sim_exp.diag()
        
        loss_per_anchor = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8) + 1e-8)
        valid_anchors = (pos_sim > 0).float()
        count = torch.sum(valid_anchors)
        total_loss = torch.sum(loss_per_anchor * valid_anchors)
        return (total_loss / (count + 1e-8)) * self.lambda_contrast 

    def get_beta(self):
        beta_en = self.beta_en
        beta_cn = self.beta_cn
        return beta_en, beta_cn

    # Sửa lại phương thức get_theta để sử dụng shared encoder với doc_embeddings
    def get_theta(self, bow, lang='en'):
        if isinstance(bow, np.ndarray):
            bow = torch.tensor(bow, dtype=torch.float, device=self.beta_en.device)
        elif hasattr(bow, 'device') and bow.device != self.beta_en.device:
            bow = bow.to(self.beta_en.device)
        
        # Đưa qua MLP tương ứng với ngôn ngữ
        if lang == 'en':
            bow_projected = self.mlp_en(bow)
        else:  # lang == 'cn'
            bow_projected = self.mlp_cn(bow)
        
        # Sử dụng encoder
        theta, mu, logvar = self.encoder(bow_projected)
        
        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def loss_function(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topic)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS

    def csim_theta(self, bow, doc):
        # Đảm bảo doc ở cùng thiết bị với bow
        if isinstance(doc, np.ndarray):
            doc = torch.tensor(doc, dtype=torch.float, device=bow.device)
        elif hasattr(doc, 'device') and doc.device != bow.device:
            doc = doc.to(bow.device)
            
        pbow = self.prj_rep(bow)
        pdoc = self.prj_doc(doc)
        
        # Tính toán ma trận cosine similarity
        bow_norm = pbow.norm(dim=-1, keepdim=True)
        doc_norm = pdoc.norm(dim=-1, keepdim=True)
        
        csim_matrix = torch.matmul(pbow, pdoc.T) / (torch.matmul(bow_norm, doc_norm.T) + 1e-8)
        csim_matrix = torch.exp(csim_matrix)
        csim_matrix = csim_matrix / (csim_matrix.sum(dim=1, keepdim=True) + 1e-8)
        
        return -csim_matrix.log()

    def compute_loss_InfoNCE(self, rep, contextual_emb):
        if self.weight_infoncetheta <= 1e-6:
            return 0.
        else:
            # Đảm bảo contextual_emb ở cùng thiết bị với rep
            if isinstance(contextual_emb, np.ndarray):
                contextual_emb = torch.tensor(contextual_emb, dtype=torch.float, device=rep.device)
            elif hasattr(contextual_emb, 'device') and contextual_emb.device != rep.device:
                contextual_emb = contextual_emb.to(rep.device)
                
            sim_matrix = self.csim_theta(rep, contextual_emb)
            return sim_matrix.diag().mean()

    def csim(self, beta_en, beta_cn):
        # Project beta_en and beta_cn into common space
        pbeta_en = self.prj_beta_en(beta_en)  # [K, 384]
        pbeta_cn = self.prj_beta_cn(beta_cn)  # [K, 384]

        # Calculate cosine similarity matrix
        csim_matrix = (pbeta_en @ pbeta_cn.T) / (pbeta_en.norm(keepdim=True, dim=-1) @ pbeta_cn.norm(keepdim=True, dim=-1).T + 1e-8)
        
        # Convert to exponential form
        csim_matrix = torch.exp(csim_matrix)
        
        # Normalize so each row sums to 1
        csim_matrix = csim_matrix / (csim_matrix.sum(dim=1, keepdim=True) + 1e-8)
        
        # Return -log of probability matrix
        return -csim_matrix.log()

    def InfoNce(self, beta_en, beta_cn):
        # Calculate -log(p) matrix
        log_p_matrix = self.csim(beta_en, beta_cn)
        loss = log_p_matrix.diag().mean()
        return loss
        
    # Sửa lại phương thức forward để sử dụng doc_embeddings
    def forward(self,x_en, x_cn, document_info=None, cluster_info=None):

        # Lấy doc embeddings (vẫn giữ cho InfoNCE loss)
        doc_embeddings_en = document_info.get('doc_embedding_en')
        doc_embeddings_cn = document_info.get('doc_embedding_cn')

        # Lấy topic distributions từ BOW thông qua MLPs và encoder
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')
        # Lấy beta matrices
        beta_en, beta_cn = self.get_beta()
        
        loss = 0.
        tmp_rst_dict = dict()

        # Reconstruct BOW từ theta (vẫn giữ BOW reconstruction loss)
        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        
        # Tính loss
        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn
        tmp_rst_dict['loss_en'] = loss_en
        tmp_rst_dict['loss_cn'] = loss_cn

        # Contrastive loss
        loss_contrastive = 0.0
        if cluster_info and 'cluster_en' in cluster_info and 'cluster_cn' in cluster_info:
            cluster_info_en = cluster_info['cluster_en']
            cluster_info_cn = cluster_info['cluster_cn']
            if cluster_info_en is not None and cluster_info_cn is not None:
                loss_contrastive = self.contrastive_loss(theta_en, theta_cn, cluster_info_en, cluster_info_cn)
                loss += loss_contrastive

        # InfoNCE loss for beta matrices
        infonce = (self.InfoNce(beta_en, beta_cn) + self.InfoNce(beta_cn,beta_en)) * self.infonce_alpha /2
        loss += infonce

        loss_InfoNCEtheta = 0.0
        if doc_embeddings_en is not None and doc_embeddings_cn is not None:
            loss_InfoNCEtheta = self.compute_loss_InfoNCE(theta_en, doc_embeddings_en) + self.compute_loss_InfoNCE(theta_cn, doc_embeddings_cn)
            loss_InfoNCEtheta *= self.weight_infoncetheta
            loss += loss_InfoNCEtheta   

        # Lưu loss components
        tmp_rst_dict['loss_contrastive'] = loss_contrastive
        tmp_rst_dict['loss_infonce'] = infonce   
        tmp_rst_dict['loss_infoncetheta'] = loss_InfoNCEtheta
        
        rst_dict = {
            'loss': loss,
        }

        rst_dict.update(tmp_rst_dict)

        return rst_dict