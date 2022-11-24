import torch
import torch.nn as nn

from .hi_encoder import PretrainingEncoder


# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('weights initialization finished!')

class HiCo(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(HiCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
  
        print(" moco parameters",K,m,T)

        self.G = args_encoder['granularity']

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # clip level queue
        self.register_buffer("c_queue", torch.randn(dim, self.G*K))
        self.c_queue = nn.functional.normalize(self.c_queue, dim=0)
        self.register_buffer("c_queue_ptr", torch.zeros(1, dtype=torch.long))

        # part level queue
        self.register_buffer("p_queue", torch.randn(dim, self.G*K))
        self.p_queue = nn.functional.normalize(self.p_queue, dim=0)
        self.register_buffer("p_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # domain level queues
        # temporal domain queue
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("i_queue", torch.randn(dim, K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, c_keys, p_keys, t_keys, s_keys, i_keys):
        N,L,C = c_keys.shape

        assert self.K % N == 0  # for simplicity

        c_ptr = int(self.c_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.c_queue[:, c_ptr:c_ptr + L*N] = c_keys.reshape(N*L,-1).T
        c_ptr = (c_ptr + L*N) % self.K  # move pointer
        self.c_queue_ptr[0] = c_ptr
        
        p_ptr = int(self.p_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.p_queue[:, p_ptr:p_ptr + L*N] = p_keys.reshape(N*L,-1).T
        p_ptr = (p_ptr + L*N) % self.K  # move pointer
        self.p_queue_ptr[0] = p_ptr

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K  # move pointer
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K  # move pointer
        self.s_queue_ptr[0] = s_ptr

        i_ptr = int(self.i_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
        i_ptr = (i_ptr + N) % self.K  # move pointer
        self.i_queue_ptr[0] = i_ptr

    def forward(self, qc_input, qp_input, kc_input, kp_input):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """

        # compute clip level, part level, temporal domain level, spatial domain level and instance level features
        qc, qp, qt, qs, qi = self.encoder_q(qc_input, qp_input)  # queries: NxC
        
        qc = nn.functional.normalize(qc, dim=2)
        qc0 = qc[:,0] # anchor feature
        qp = nn.functional.normalize(qp, dim=2)
        qp0 = qp[:,0] # anchor feature
        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qi = nn.functional.normalize(qi, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            kc, kp, kt, ks, ki = self.encoder_k(kc_input, kp_input)	  # keys: NxC
    
            kc = nn.functional.normalize(kc, dim=2)
            kp = nn.functional.normalize(kp, dim=2)
            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ki = nn.functional.normalize(ki, dim=1)

        N, G, C = kc.shape
     
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: NxL 
        # G: number of other granularities
        l_pos_c = torch.einsum('nc,nlc->nl', [qc0, kc])
        l_pos_p = torch.einsum('nc,nlc->nl', [qp0, kp])
        l_pos_t = torch.einsum('nc,nc->n', [qt, ks]).unsqueeze(1)
        l_pos_s = torch.einsum('nc,nc->n', [qs, kt]).unsqueeze(1)
        l_pos_i = torch.einsum('nc,nc->n', [qi, ki]).unsqueeze(1)

        # negative logits: NxK
        l_neg_c = torch.einsum('nc,ck->nk', [qc0, self.c_queue.clone().detach()])
        l_neg_p = torch.einsum('nc,ck->nk', [qp0, self.p_queue.clone().detach()])
        l_neg_t = torch.einsum('nc,ck->nk', [qt, self.s_queue.clone().detach()])
        l_neg_s = torch.einsum('nc,ck->nk', [qs, self.t_queue.clone().detach()])
        l_neg_i = torch.einsum('nc,ck->nk', [qi, self.i_queue.clone().detach()])

        # logits: Nx(1+K)
        logits_c = torch.cat([l_pos_c, l_neg_c], dim=1)
        logits_p = torch.cat([l_pos_p, l_neg_p], dim=1)
        logits_t = torch.cat([l_pos_t, l_neg_t], dim=1)
        logits_s = torch.cat([l_pos_s, l_neg_s], dim=1)
        logits_i = torch.cat([l_pos_i, l_neg_i], dim=1)

        # apply temperature
        logits_c /= self.T
        logits_p /= self.T
        logits_t /= self.T
        logits_s /= self.T
        logits_i /= self.T


        # positive key indicators
        mask_c = torch.cat([torch.ones(N,G),torch.zeros(N,G*self.K)], dim=1).cuda()
        mask_p = torch.cat([torch.ones(N,G),torch.zeros(N,G*self.K)], dim=1).cuda()
        labels_t = torch.zeros(logits_t.shape[0], dtype=torch.long).cuda()
        labels_s = torch.zeros(logits_s.shape[0], dtype=torch.long).cuda()
        labels_i = torch.zeros(logits_i.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(kc, kp, kt, ks, ki)

        return logits_c, logits_p, logits_t, logits_s, logits_i, \
                mask_c, mask_p, labels_t, labels_s, labels_i,