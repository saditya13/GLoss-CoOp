import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # Store BPE vocabulary embeddings for end-of-training analysis
        self.register_buffer("vocab_embeddings", clip_model.token_embedding.weight.detach().type(dtype))

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def print_learned_prompts_vocab_matches(self, batch_idx=0):
        """
        Compute and print the top-1 nearest BPE vocabulary token for each learned prompt token.
        
        Compares learned context embeddings against the original BPE vocabulary embeddings
        to show what natural language tokens the learned prompts are most similar to.
        
        Args:
            batch_idx (int): Index of the batch for class-specific contexts. Default is 0.
        """
        prompt_ctx = self.ctx.detach().float()
        if prompt_ctx.dim() == 3:
            prompt_ctx = prompt_ctx[batch_idx]
        
        # --- Debug ---
        # print(f"ctx dtype: {prompt_ctx.dtype}")
        # print(f"ctx shape: {prompt_ctx.shape}")
        # print(f"ctx norms: {prompt_ctx.norm(dim=-1)}")
        # print(f"ctx has nan: {torch.isnan(prompt_ctx).any()}")
        # print(f"ctx has inf: {torch.isinf(prompt_ctx).any()}")
        # print(f"ctx min/max: {prompt_ctx.min():.4f} / {prompt_ctx.max():.4f}")
        # -------------

        vocab_emb = self.vocab_embeddings.detach().float()
        
        # Normalize embeddings for cosine similarity
        prompt_norm = prompt_ctx / prompt_ctx.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vocab_norm = vocab_emb / vocab_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        
        # Compute cosine similarity between each learned prompt token and all vocab tokens
        similarities = prompt_norm @ vocab_norm.t()  # (n_ctx, vocab_size)
        
        # Get top-1 matches
        top_values, top_indices = similarities.topk(1, dim=-1)
        
        # Print results
        print("\n" + "="*60)
        print("Learned Prompts - Top-1 BPE Vocabulary Matches")
        print("="*60)
        for idx, (sim, token_id) in enumerate(zip(top_values.squeeze(-1).tolist(), top_indices.squeeze(-1).tolist())):
            token_text = _tokenizer.decode([token_id])
            print(f"Prompt Token {idx}: '{token_text}' (ID: {token_id}, Similarity: {sim:.4f})")
        print("="*60 + "\n")

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def encode_text_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def forward(self, image):
        image_features = self.encode_image(image)
        text_features = self.encode_text_features()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward_with_label_graph(self, image, labels):
        image_features = self.encode_image(image)
        text_features = self.encode_text_features()
        label_features = text_features[labels]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        label_features = label_features / label_features.norm(dim=-1, keepdim=True)

        # concat_features = torch.cat([image_features, label_features], dim=-1)
        # # concat_features = image_features * label_features  # element-wise product
        # concat_features = concat_features / concat_features.norm(dim=-1, keepdim=True)

        alpha = 1.0  # tunable — amplifies prompt's influence on graph
        concat_features = torch.cat([image_features, alpha * label_features], dim=-1)
        concat_features = concat_features / concat_features.norm(dim=-1, keepdim=True)
        
        return concat_features

    

@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.COOP.LOSS_TYPE in ["cross_entropy", "gloss","scl", "ce+gloss", "ce+scl"]
    
  
    def after_epoch(self):
        if self.cfg.TRAINER.COOP.LOSS_TYPE == "ce+gloss":
            if len(self._epoch_batch_losses["ce"]) > 0:
                self.ce_losses.append(np.mean(self._epoch_batch_losses["ce"]))
                self.g_losses.append(np.mean(self._epoch_batch_losses["g"]))
                self.total_losses.append(np.mean(self._epoch_batch_losses["total"]))
                print(f"[Epoch {self.epoch}] CE: {self.ce_losses[-1]:.4f} | "
                    f"GLoss: {self.g_losses[-1]:.4f} | "
                    f"Total: {self.total_losses[-1]:.4f}"
                    f"Number of batch losses: {len(self._epoch_batch_losses['ce'])}")
                # Reset for next epoch
                self._epoch_batch_losses = {"ce": [], "g": [], "total": []}

        super().after_epoch()
    
    def after_train(self):
        if self.cfg.TRAINER.COOP.LOSS_TYPE == "ce+gloss":
            # print(f"[DEBUG] Collected {len(self.ce_losses)} epoch losses: {self.ce_losses}")
            self._plot_losses()

        super().after_train()
        print("\n[Final] Learned prompt vocab matches:")
        self.model.prompt_learner.print_learned_prompts_vocab_matches()
        

    def _plot_losses(self):
        epochs = range(1, len(self.ce_losses) + 1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.ce_losses,    label="CE Loss")
        plt.plot(epochs, self.g_losses,     label="GLoss")
        plt.plot(epochs, self.total_losses, label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = osp.join(self.output_dir, "loss_curves_over_epochs.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def gaussian_similarity(self, emb, sigma):
        """Compute Gaussian kernel weights."""
        sq_dists = torch.cdist(emb, emb, p=2) ** 2 
        weight = torch.exp(-sq_dists / (2*(sigma**2)))
        weight = weight - torch.diag(torch.diag(weight))
        return weight

    def normalize_adj(self, adj: torch.Tensor):
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degrees = adj.sum(dim=1)
        degrees = degrees.clamp(min=1e-6)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    def plot_graph(self, adj):
        graph_dir = osp.join(self.output_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.imshow(adj.detach().cpu().numpy(), cmap="coolwarm", vmin=0.0, vmax=1.0)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Adjacency matrix")
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.tight_layout()
        plt.savefig(osp.join(graph_dir, "adj_mat.png"), dpi=300)
        plt.close()

    def gloss_lpa(self, train_emb, test_emb, Ytrain, sigma, num_labels):
        device = train_emb.device
        emb = torch.cat((train_emb, test_emb), dim=0)
        num_nodes = emb.shape[0]

        labels = torch.cat(
            [Ytrain, torch.zeros(test_emb.shape[0], dtype=Ytrain.dtype, device=device)],
            dim=0,
        )

        Y = torch.zeros((num_nodes, num_labels), dtype=torch.float32, device=device)
        for k in range(num_labels):
            Y[labels == k, k] = 1.0

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[: Ytrain.shape[0]] = True

        # emb = emb / emb.norm(dim=1, keepdim=True)
        # if torch.isnan(emb).any() or torch.isinf(emb).any():
        #     raise ValueError("NaN or inf in embeddings")

        adj = self.gaussian_similarity(emb, sigma).to(torch.float32) 
        self.plot_graph(adj)
        adj = adj + adj.t()
        adj_norm = self.normalize_adj(adj).to_dense()
        print(f"Adjacency matrix stats: min={adj_norm.min().item():.4f}, max={adj_norm.max().item():.4f}, mean={adj_norm.mean().item():.4f}, std={adj_norm.std().item():.4f}")

        # Tran = adj_norm / adj_norm.sum(dim=0, keepdim=True)
        # row_sum = Tran.sum(dim=1, keepdim=True)
        # T = Tran / row_sum

        T = adj_norm / adj_norm.sum(dim=1, keepdim=True).clamp(min=1e-6)
        N_l = train_emb.shape[0]
        T_ul = T[N_l:, :N_l]
        T_uu = T[N_l:, N_l:]

        I = torch.eye(T_uu.shape[0], dtype=torch.float32, device=device)
        F_UU = torch.linalg.solve(I - T_uu, T_ul.mm(Y[train_mask]))

        if torch.isnan(F_UU).any() or torch.isinf(F_UU).any():
            raise ValueError("NaN or inf in F_UU before normalization")

        return F_UU
    def compute_gloss(self, concat_features, labels, sigma, gamma):
        # self.gloss_temp = nn.Parameter(torch.tensor(1.0))
        # print("Computing GLoss ...")
        # print(f"GLoss parameters: sigma={sigma}, gamma={gamma}")
        # print(f"concat_features shape: {concat_features.shape}")
        mask1 = torch.randperm(concat_features.size(0)) < concat_features.size(0) * gamma
        mask2 = ~mask1
        emb_lab_set = concat_features[mask1]
        emb_eval_set = concat_features[mask2]
        labels_lab_set = labels[mask1]
        labels_eval_set = labels[mask2]
        pred = self.gloss_lpa(
            emb_lab_set,
            emb_eval_set,
            labels_lab_set,
            sigma,
            self.model.prompt_learner.n_cls)
        loss = F.cross_entropy(pred, labels_eval_set)
        return loss
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.ce_losses    = []
        self.g_losses     = []
        self.total_losses = []
        self._epoch_batch_losses = {"ce": [], "g": [], "total": []}

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        loss_type = self.cfg.TRAINER.COOP.LOSS_TYPE
        sigma = self.cfg.TRAINER.COOP.GLOSS_SIGMA
        gamma = self.cfg.TRAINER.COOP.GLOSS_GAMMA
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if loss_type == "cross_entropy":
                output = self.model(image)
                loss = F.cross_entropy(output, label)

            elif loss_type == "gloss":
                concat_emb = self.model.forward_with_label_graph(image, label)
                loss = self.compute_gloss(concat_emb, label, sigma, gamma)

            elif loss_type == "scl":
                concat_emb = self.model.forward_with_label_graph(image, label)
                loss = SupConLoss(temperature=0.07)(concat_emb.unsqueeze(1), labels=label)

            elif loss_type == "ce+gloss":
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                concat_emb = self.model.forward_with_label_graph(image, label)
                g_loss = self.compute_gloss(concat_emb, label, sigma, gamma)
                loss = 0.3 * ce_loss + 0.7 * g_loss
                self._epoch_batch_losses["ce"].append(ce_loss.item())
                self._epoch_batch_losses["g"].append(g_loss.item())
                self._epoch_batch_losses["total"].append(loss.item())

            elif loss_type == "ce+scl":
                output = self.model(image)
                ce_loss = F.cross_entropy(output, label)
                concat_emb = self.model.forward_with_label_graph(image, label)
                scl_loss = SupConLoss(temperature=0.07)(concat_emb.unsqueeze(1), labels=label)
                loss = 0.3 * ce_loss + 0.7 * scl_loss

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            self.model_backward_and_update(loss)
        # for name, param in self.model.prompt_learner.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad_norm = {param.grad.norm().item():.6f}") 
        
        # if loss_type == "gloss":
        #     loss_summary = {"loss": loss.item()}
        # else: 
        #     loss_summary = {
        #     "loss": loss.item(),
        #     "acc": compute_accuracy(output, label)[0].item(),
        # }
            loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss