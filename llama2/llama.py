from dataclasses import dataclass
from  typing import Optional,Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math
import inspect
from config import ModelArgs

'''
RMSNorm 输入[N,M],输出[N,M]

rsqrt函数: 对每个input的平方根的倒数
a=torch.Tensor([1,4,9,16,25])
torch.rsqrt(a)
tensor([1.0000, 0.5000, 0.3333, 0.2500, 0.2000])

a=torch.Tensor([[1,2],[4,6]])
a.mean(-1, keepdim=True) 
tensor([[1.5000],
        [5.0000]])
keepdim 保持数据维度
'''
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

'''
freqs_cos 维度: [end, head_dim//2]
freqs_sin维度: [end, head_dim//2]
'''
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

'''
扩散fres_cis维度
fres_cis原维度:  [max_seq_length, head_dim//2]
x:维度 [batch, max_seq_length, local_heads, head_dim//2]
输出:  fres_cis维度: [1, max_seq_length, 1,head_dim//2]
'''
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

'''
计算x, k加入旋转位置:
reshape(xq.shape[:-1] + (-1, 2))  数据维度调整
如果x.shape =(2,4,8,8) 则 改为 (2,4,8,4,2)
unbind(-1) 最后一个维度改成复数返回 实数 维度:(2,4,8,4),虚数维度(2,4,8,4)

torch.stack要求数据维度相同
a.shape=[4,2]
torch.stack([a,a],dim=-1) 得到数据维度  [4,2,2]
torch.stack([a,a]) 得到数据维度  [2,4,2]

flatten捋平
a.shape=[2,4,2]
a.flatten(1) 得到数据维度:  [2,8]
'''
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

'''
None 新增一个维度
x.shape=[3,4]
x[:,None,:]得到  [3,1,4]

expand: 维度扩张

'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


'''
torch.triu 将矩阵转成对角矩阵
a=torch.full((1, 1,10, 10),float("-inf"))
tensor([[[[-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]]]])

torch.triu(a, diagonal=1)
tensor([[[[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
          [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
          [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
          [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
          [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
          [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])
'''
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        #transpose 跟换两个维度
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            ## (bs, n_local_heads, seqlen, head_dim)@(bs, n_local_heads, head_dim, seqlen)  得到  (bs, n_local_heads, seqlen, seqlen)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            #使得前面的token在后面的token计算attention时得分为0, Masked Self-Attention (masked.png)，由于根据softmax算法,负无穷的score会是0(e的负无穷次方是0). 
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            # dim对最后一个维度的值求softmax, 
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # scores  (bs, n_local_heads, seqlen, seqlen) @  (bs, n_local_heads, seqlen, head_dim) => (bs, n_local_heads, seqlen, head_dim)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        #transpose之后需要view，需要使用contiguous使得数据内存连续.
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # h[:, [-1], :]  [-1]取最后一个数据,  h维度:[3,4,4], h[:,[-1],:] 得到 [3,1,4] 这个1是最后一个数据  
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None
        return logits


    #@torch.inference_mode()
    @torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # 去除一维数据  logits:[3,1,4]， logits[:,-1,:]得到  [3,4]
            logits = logits[:, -1, :] 
            if temperature == 0.0:
                # topk返回值(values, indexes)
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next==eos:
                break

        return idx

    #配置衰减参数,dim>= 2使用衰减参数，否则不使用
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for pn,p in param_dict.items()if p.dim()>=2]
        nodecay_params = [p for pn,p in param_dict.items()if p.dim()< 2]
        optim_groups = [
            {"params":decay_params , 'weight_decay':weight_decay},
            {"params":nodecay_params , 'weight_decay':0.0 },
        ]
        fuse_available = "fuse" in inspect.signature(torch.optim.AdamW).parameters
        use_fuse = fuse_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fuse else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,**extra_args)
        optimizer.param_groups
        return optimizer

    def export(self, filepath='model.bin'):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, 'wb')

        def serialize(t):
            d = t.detach().cpu().view(-1).numpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # first write out the header
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                       n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # next write out the embedding weights
        serialize(self.tok_embeddings.weight)

        # now all the layers
        # attention weights
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        # ffn weights
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize(self.norm.weight)
        # note: no need to write final classifier weights due to weight sharing
        # freqs_cis
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        # write to binary file
        f.close()
        print(f"wrote {filepath}")

    # 评估模型GPU利用率MFU
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

'''
for pn,p in model.named_parameters():
        print(pn)

tok_embeddings.weight
layers.0.attention.wq.weight
layers.0.attention.wk.weight
layers.0.attention.wv.weight
layers.0.attention.wo.weight
layers.0.feed_forward.w1.weight
layers.0.feed_forward.w2.weight
layers.0.feed_forward.w3.weight
layers.0.attention_norm.weight
layers.0.ffn_norm.weight
layers.1.attention.wq.weight
layers.1.attention.wk.weight
layers.1.attention.wv.weight
layers.1.attention.wo.weight
layers.1.feed_forward.w1.weight
layers.1.feed_forward.w2.weight
layers.1.feed_forward.w3.weight
layers.1.attention_norm.weight
layers.1.ffn_norm.weight
layers.2.attention.wq.weight
layers.2.attention.wk.weight
layers.2.attention.wv.weight
layers.2.attention.wo.weight
layers.2.feed_forward.w1.weight
layers.2.feed_forward.w2.weight
layers.2.feed_forward.w3.weight
layers.2.attention_norm.weight
layers.2.ffn_norm.weight
layers.3.attention.wq.weight
layers.3.attention.wk.weight
layers.3.attention.wv.weight
layers.3.attention.wo.weight
layers.3.feed_forward.w1.weight
layers.3.feed_forward.w2.weight
layers.3.feed_forward.w3.weight
layers.3.attention_norm.weight
layers.3.ffn_norm.weight
layers.4.attention.wq.weight
layers.4.attention.wk.weight
layers.4.attention.wv.weight
layers.4.attention.wo.weight
layers.4.feed_forward.w1.weight
layers.4.feed_forward.w2.weight
layers.4.feed_forward.w3.weight
layers.4.attention_norm.weight
layers.4.ffn_norm.weight
layers.5.attention.wq.weight
layers.5.attention.wk.weight
layers.5.attention.wv.weight
layers.5.attention.wo.weight
layers.5.feed_forward.w1.weight
layers.5.feed_forward.w2.weight
layers.5.feed_forward.w3.weight
layers.5.attention_norm.weight
layers.5.ffn_norm.weight
layers.6.attention.wq.weight
layers.6.attention.wk.weight
layers.6.attention.wv.weight
layers.6.attention.wo.weight
layers.6.feed_forward.w1.weight
layers.6.feed_forward.w2.weight
layers.6.feed_forward.w3.weight
layers.6.attention_norm.weight
layers.6.ffn_norm.weight
layers.7.attention.wq.weight
layers.7.attention.wk.weight
layers.7.attention.wv.weight
layers.7.attention.wo.weight
layers.7.feed_forward.w1.weight
layers.7.feed_forward.w2.weight
layers.7.feed_forward.w3.weight
layers.7.attention_norm.weight
layers.7.ffn_norm.weight
norm.weight
'''