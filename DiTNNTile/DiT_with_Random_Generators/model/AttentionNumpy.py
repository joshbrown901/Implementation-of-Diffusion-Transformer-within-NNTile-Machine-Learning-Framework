import numpy as np

def softmax_np(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

class Attention_numpy:
    def __init__(self, config):
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads * self.head_dim

        self.qkv_weight = np.zeros((3 * self.att_dim, self.hidden_size), dtype=np.float64)
        self.qkv_bias   = np.zeros((3 * self.att_dim,), dtype=np.float64)
        self.out_weight = np.zeros((self.hidden_size, self.att_dim), dtype=np.float64)
        self.out_bias   = np.zeros((self.hidden_size,), dtype=np.float64)

    def forward(self, x):
        x = x.astype(np.float64)
        B, N, _ = x.shape

        self.x = x  # cache input for backward
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q, k, v = np.split(qkv, 3, axis=-1)

        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(0,1,3,2)) * scale
        att = softmax_np(scores)

        out = att @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.att_dim)

        self.q, self.k, self.v = q, k, v  # cache
        self.att, self.scores, self.scale = att, scores, scale  # cache

        out_final = out @ self.out_weight.T + self.out_bias
        self.out_before_linear = out  # cache for backward
        return out_final

    def backward(self, dout):
        B, N, _ = dout.shape

        # Grad w.r.t. out_weight and out_bias
        d_out_weight = dout.reshape(-1, self.hidden_size).T @ self.out_before_linear.reshape(-1, self.att_dim)
        d_out_bias   = np.sum(dout, axis=(0, 1))

        # Grad w.r.t. out (before final linear)
        d_out = dout @ self.out_weight

        # Backprop through reshape + transpose
        d_out = d_out.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Grad w.r.t. att and v
        d_att = d_out @ self.v.transpose(0, 1, 3, 2)
        d_v = self.att.transpose(0, 1, 3, 2) @ d_out

        # Grad w.r.t. softmax scores
        d_scores = d_att * self.att
        d_scores -= self.att * np.sum(d_att * self.att, axis=-1, keepdims=True)

        d_scores *= self.scale

        # Grad w.r.t. q and k
        d_q = d_scores @ self.k
        d_k = d_scores.transpose(0, 1, 3, 2) @ self.q

        # Backprop through reshape + transpose
        d_q = d_q.transpose(0, 2, 1, 3).reshape(B, N, -1)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(B, N, -1)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(B, N, -1)

        # Grad w.r.t. qkv linear layer
        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1)

        d_qkv_weight = d_qkv.reshape(-1, 3 * self.att_dim).T @ self.x.reshape(-1, self.hidden_size)
        d_qkv_bias   = np.sum(d_qkv, axis=(0, 1))

        d_x = d_qkv @ self.qkv_weight

        grads = {
            'qkv_weight': d_qkv_weight,
            'qkv_bias': d_qkv_bias,
            'out_weight': d_out_weight,
            'out_bias': d_out_bias
        }

        return d_x, grads