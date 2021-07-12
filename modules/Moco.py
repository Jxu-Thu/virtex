import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, intra_dim=128,inter_dim=1024, hidden_dim = 2048,K=32768, m=0.999, T=0.07,config = None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        if config is not None:
            self.encoder_q = base_encoder(config)
            self.encoder_k = base_encoder(config)
            print(self.encoder_q.config.hidden_size)
            dim_mlp = self.encoder_q.config.hidden_size
        else:
            self.encoder_q = base_encoder()
            self.encoder_k = base_encoder()
            dim_mlp = self.encoder_q.block.expansion*512

        
        self.mlp_q_intra = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, intra_dim))
        self.mlp_q_inter = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, inter_dim))
        self.mlp_k_intra = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, intra_dim))
        self.mlp_k_inter = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, inter_dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.mlp_q_intra.parameters(), self.mlp_k_intra.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.mlp_q_inter.parameters(), self.mlp_k_inter.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_intra", torch.randn(intra_dim, K))
        self.queue_intra = nn.functional.normalize(self.queue_intra, dim=0)
        self.register_buffer("queue_inter", torch.randn(inter_dim, K))
        self.queue_inter = nn.functional.normalize(self.queue_inter, dim=0)

        self.register_buffer("queue_intra_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_inter_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.mlp_q_intra.parameters(), self.mlp_k_intra.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.mlp_q_inter.parameters(), self.mlp_k_inter.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, mode="intra"):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        queue_ptr = getattr(self, f"queue_{mode}_ptr")
        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue = getattr(self, f"queue_{mode}")
        queue[:, ptr:ptr + batch_size] = keys.T
        # self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        # self.queue_ptr[0] = ptr
        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q=None, im_k=None, text_mask_q = None, text_mask_k = None, mode = "intra"):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if mode == "intra":
            if text_mask_q is not None:
                q = self.encoder_q(im_q, text_mask_q).pooler_output
            else:
                q = self.encoder_q(im_q)  # queries: NxC
            q = self.mlp_q_intra(q)    
            q = nn.functional.normalize(q, dim=1)
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                if text_mask_k is not None:
                    k = self.encoder_k(im_k, text_mask_k).pooler_output  # keys: NxC
                else:
                    k = self.encoder_k(im_k)
                k = self.mlp_k_intra(k)
                k = nn.functional.normalize(k, dim=1)

            # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            
            # print(self.queue_intra.shape)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_intra.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            return logits, labels
        elif mode == "q_inter":
            if text_mask_q is not None:
                q = self.encoder_q(im_q, text_mask_q).pooler_output
            else:
                q = self.encoder_q(im_q)  # queries: NxC
            q = self.mlp_q_inter(q)    
            q = nn.functional.normalize(q, dim=1)
            return q
        elif mode == "k_inter":
            with torch.no_grad():
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                if text_mask_k is not None:
                    k = self.encoder_k(im_k, text_mask_k).pooler_output  # keys: NxC
                else:
                    k = self.encoder_k(im_k)
                k = self.mlp_k_inter(k)
                k = nn.functional.normalize(k, dim=1)
                return k

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output