from functools import reduce
# from utils import nice_print, mem_report, cpu_stats
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F


class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau, FourDlayer):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        
        for i in range(num_layers*FourDlayer):
            # print("==============",num_layers)
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # print(input.shape)
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        # print("Seq Len: ", input.size(0))
        c_history_states1 = []
        h_states1 = []
        outputs1 = []
        top_h_rep1=[]
        c_history_states2 = []
        h_states2 = []
        outputs2 = []
        top_h_rep2=[]
        # i=0
        # for step, x in enumerate(input):
        #     print("Step", step)
        #     print("X", x.shape)

        for step, x in enumerate(input):
            # print(step)
            # print(i)
            # i+=1
            # print("Step: ", step)
            x1=x[:,:,0:32,:,:]
            x2=x[:,:,32:64,:,:]
            # print("xshape",x.shape)
            

            for cell_idx, cell in enumerate(self._cells):
                if cell_idx==4:
                    break
                if step == 0:
                    c_history1, m1, h1 = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history2, m2, h2 = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history_states1.append(c_history1)
                    c_history_states2.append(c_history2)

                    h_states1.append(h1)
                    h_states2.append(h2)
                    # print(len(h_states))
                    
#                 print("X1 shape: ",x1.shape )
#                 print("H1 shape: ",h_states1[cell_idx].shape)
                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history1, m1, h1 = cell(
                    x1, c_history_states1[cell_idx], m1, h_states1[cell_idx]
                )
                c_history2, m2, h2 = cell(
                    x2, c_history_states2[cell_idx], m2, h_states2[cell_idx]
                )
                
                c_history_states1[cell_idx] = c_history1
                h_states1[cell_idx] = h1

                c_history_states2[cell_idx] = c_history2
                h_states2[cell_idx] = h2
                
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x1 = h1
                x2 = h2
                if cell_idx==3:
                    # print("h1 shape: ", h1.shape)
                    top_h_rep1.append(h1)
                    top_h_rep2.append(h2)
                

            # print()
            outputs1.append(h1)
            # print("Hidden Rep: ",h.shape)
            # print("outputs: ",len(outputs))
        # print("OUT: ",torch.cat(outputs, dim=1).shape)
        # l=torch.cat(outputs, dim=1)
        # print((l.shape))
        # Concat along the channels
        top_h_tensor1=torch.cat(top_h_rep1, dim=2)
        top_h_tensor2=torch.cat(top_h_rep2, dim=2)

        top_h_tensor1=top_h_tensor1.squeeze(0)
        top_h_tensor2=top_h_tensor2.squeeze(0)
        # top_h_tensor=top_h_tensor.view(64,-1)
        # print(top_h_tensor.shape)
        # TODO Add a classifier
        top_h_tensor=torch.cat((top_h_tensor1,top_h_tensor2),1)
#         print("top_h_tensor: ",top_h_tensor.shape)

        return top_h_tensor
    

class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall = self.self_attention_fast(r, c_history)
        # nice_print(**locals())
        # mem_report()
        # cpu_stats()

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # # nice_print(**locals())
        # print("M: ",m.shape)
        # print("H: ",h.shape)
        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)
        # print("Init c_history:", c_history.shape)
        # print("Init H:", h.shape)
        # print("Init M:", m.shape)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")