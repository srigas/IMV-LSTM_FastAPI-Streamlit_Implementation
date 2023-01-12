import torch
from torch import nn

# Adapted from Alexey Kurochkin's version
# VI stands for Variable importance
# VWTI stands for Variable-wise temportal importance
class IMVLSTM(torch.jit.ScriptModule):
    # the input dim is fixed by our choice of features
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        # Input-to-hidden transition tensor elements
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        # Hidden-to-hidden transition tensor elements
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        # Biases
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        # f_n
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        # phi_n for the linear mapping
        self.Phi = nn.Linear(2*n_units, output_dim)
        # Input dim and hidden units
        self.n_units = n_units
        self.input_dim = input_dim
    
    @torch.jit.script_method
    def forward(self, x):
        # Initialization of \tilde{h}_t and cells
        h_tilde_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        c_tilde_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # Note that einsum = Einstein's summation notation, used here for tensor multiplications
            j_tilde_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilde_t, self.W_j) + torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1),self.U_j) + self.b_j)
            i_tilde_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilde_t, self.W_i) + torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilde_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilde_t, self.W_f) + torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilde_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilde_t, self.W_o) + torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            c_tilde_t = c_tilde_t*f_tilde_t + i_tilde_t*j_tilde_t
            h_tilde_t = (o_tilde_t*torch.tanh(c_tilde_t))
            outputs += [h_tilde_t]
        # Stack the hidden matrix results
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # Calculation of variable-wise temporal importance coefficients, alpha
        # Numerator of alpha
        VWTI = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        VWTI = torch.exp(VWTI) # add the exponential
        # Denominator of alpha
        VWTI = VWTI/torch.sum(VWTI, dim=1, keepdim=True)
        # Calculation of importance coefficients
        g_n = torch.sum(VWTI*outputs, dim=1)
        hg = torch.cat([g_n, h_tilde_t], dim=2)
        mu = self.Phi(hg)
        VI = torch.tanh(self.F_beta(hg))
        VI = torch.exp(VI)
        VI = VI/torch.sum(VI, dim=1, keepdim=True)
        # make the final prediction
        predval = torch.sum(VI*mu, dim=1)
        
        return predval, VWTI, VI