import torch
from model import RNNModel
import numpy as np

def lstm_forward(model, inputs):
    """
    Unmodified re-implementation of LSTM forward function 
    """
    with torch.no_grad():
        weights = model.state_dict()
        embeddings = model.encoder(inputs)
        tokens, _, hidden_size = embeddings.shape

        # initialize to 0s
        h_t = torch.zeros((2, 1, hidden_size))
        c_t = torch.zeros((2, 1, hidden_size))
        h_t_minus_1 =  torch.zeros((2, 1, hidden_size))
        c_t_minus_1 = torch.zeros((2, 1, hidden_size))
        output = []

        # iterate through tokens to process sequentially
        for t in range(tokens):
            for layer in range(2):
                # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                w_ii, w_if, w_ig, w_io = np.split(weights[f"rnn.weight_ih_l{layer}"], 4, 0)
                w_hi, w_hf, w_hg, w_ho = np.split(weights[f"rnn.weight_hh_l{layer}"], 4, 0)
                b_ii, b_if, b_ig, b_io = np.split(weights[f"rnn.bias_ih_l{layer}"], 4, 0)
                b_hi, b_hf, b_hg, b_ho =  np.split(weights[f"rnn.bias_hh_l{layer}"], 4, 0)

                if layer == 0:
                    x = embeddings[t]
                else:
                    x = h
                h = h_t_minus_1[layer]
                c = c_t_minus_1[layer]

                i = x @ torch.t(w_ii)
                i += h @ torch.t(w_hi)
                i += b_hi + b_ii
                i = torch.sigmoid(i)

                f = x @ torch.t(w_if)
                f += h @ torch.t(w_hf)
                f += b_if + b_hf
                f = torch.sigmoid(f)                
                
                g = x @ torch.t(w_ig)
                g += h @ torch.t(w_hg)
                g += b_ig + b_hg
                g = torch.tanh(g)

                o = x @ torch.t(w_io)
                o += h @ torch.t(w_ho)
                o += b_io + b_ho
                o = torch.sigmoid(o)

                c = f * c + i * g
                h = o * torch.tanh(c)

                h_t[layer] = h
                c_t[layer] = c

            output.append(h)
            c_t_minus_1 = c_t
            h_t_minus_1 = h_t
            
        w_d = weights["decoder.weight"]
        w_b = weights["decoder.bias"]
        # as input, takes hidden values of final layer from all input tokens (flattened)
        output = torch.stack(output)

        flattened = output.view(output.shape[0] * output.shape[1], output.shape[2])
        z = flattened @ torch.t(w_d)
        z += w_b
        z = z.view(output.shape[0], output.shape[1], z.shape[1])
        return z
    
    
def init_gamma(model):
    """
    Initialize hidden gamma states using activations from ". <eos>" sentence
    """
    with torch.no_grad():
        init_phrase = torch.LongTensor([18, 19]).unsqueeze(1)
        embed = model.encoder(init_phrase)
        _, hidden = model.rnn(embed)
        return hidden
    

def contextual_decomposition(model : RNNModel, inputs, start, end, bias=True):
    """
    Re-implementation of LSTM forward function with contextual decomposition
    """
    with torch.no_grad():
        weights = model.state_dict()
        embeddings = model.encoder(inputs)
        tokens, _, hidden_size = embeddings.shape

        # save hidden from previous layers
        h_t_beta = torch.zeros((2, 1, hidden_size))
        h_t_gamma = torch.zeros((2, 1, hidden_size))
        c_t_beta = torch.zeros((2, 1, hidden_size))
        c_t_gamma = torch.zeros((2, 1, hidden_size))
        h_t_minus_1_beta = torch.zeros((2, 1, hidden_size))
        h_t_minus_1_gamma = torch.zeros((2, 1, hidden_size))
        c_t_minus_1_beta = torch.zeros((2, 1, hidden_size))
        c_t_minus_1_gamma = torch.zeros((2, 1, hidden_size))

        output_beta = []
        output_gamma = []

        for t in range(tokens):
            for layer in range(2):
                # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                w_ii, w_if, w_ig, w_io = np.split(weights[f"rnn.weight_ih_l{layer}"], 4, 0)
                w_hi, w_hf, w_hg, w_ho = np.split(weights[f"rnn.weight_hh_l{layer}"], 4, 0)
                b_ii, b_if, b_ig, b_io = np.split(weights[f"rnn.bias_ih_l{layer}"], 4, 0)
                b_hi, b_hf, b_hg, b_ho =  np.split(weights[f"rnn.bias_hh_l{layer}"], 4, 0)

                if layer == 0:
                    x = embeddings[t]
                    if t >= start and t < end:
                        beta_x = x
                        gamma_x = torch.zeros(x.shape)
                    else:
                        beta_x = torch.zeros(x.shape)
                        gamma_x = x
                else:
                    # use the gamma, beta from the previous layer
                    beta_x = beta_h
                    gamma_x = gamma_h

                beta_h = h_t_minus_1_beta[layer]
                gamma_h = h_t_minus_1_gamma[layer]
                beta_c = c_t_minus_1_beta[layer]
                gamma_c = c_t_minus_1_gamma[layer]

                beta_i = beta_h @ torch.t(w_hi)
                gamma_i = gamma_h @ torch.t(w_hi)

                beta_f = beta_h @ torch.t(w_hf)
                gamma_f = gamma_h @ torch.t(w_hf)

                beta_g = beta_h @ torch.t(w_hg)
                gamma_g = gamma_h @ torch.t(w_hg)

                beta_o = beta_h @ torch.t(w_ho)
                gamma_o = gamma_h @ torch.t(w_ho)

                beta_i += beta_x @ torch.t(w_ii)
                beta_f += beta_x @ torch.t(w_if)
                beta_g += beta_x @ torch.t(w_ig)
                beta_o += beta_x @ torch.t(w_io)

                gamma_i += gamma_x @ torch.t(w_ii)
                gamma_f += gamma_x @ torch.t(w_if)
                gamma_g += gamma_x @ torch.t(w_ig)
                gamma_o += gamma_x @ torch.t(w_io)

                # apply activations

                beta_i, gamma_i, bias_i = decomp_activation_three(beta_i, gamma_i, b_ii + b_hi, torch.sigmoid)
                beta_f, gamma_f, bias_f = decomp_activation_three(beta_f, gamma_f, b_if + b_hf, torch.sigmoid)
                beta_o, gamma_o, bias_o = decomp_activation_three(beta_o, gamma_o, b_io + b_ho, torch.sigmoid)
                beta_g, gamma_g, bias_g = decomp_activation_three(beta_g, gamma_g, b_ig + b_hg, torch.tanh)

                # element-wise products

                # LSTM eq. 5 (calculate next cell state)
                beta_c = beta_c * (beta_f + gamma_f + bias_f)
                gamma_c = gamma_c * (beta_f + gamma_f + bias_f)

                beta_c += beta_g * (beta_i + gamma_i + bias_i) + bias_g * beta_i
                gamma_c += gamma_g * (beta_i + gamma_i + bias_i) + bias_g * gamma_i
                if t >= start and t < end:
                    beta_c += bias_g * bias_i
                else:
                    gamma_c += bias_g * bias_i

                # LSTM eq. 6 (output gate)
                beta_ht, gamma_ht = decomp_activation_two(beta_c, gamma_c, torch.tanh)
                
                beta_h = beta_ht * (beta_o + gamma_o + bias_o)
                gamma_h = gamma_ht * (beta_o + gamma_o + bias_o)

                h_t_beta[layer] = beta_h
                h_t_gamma[layer] = gamma_h
                c_t_beta[layer] = beta_c
                c_t_gamma[layer] = gamma_c
  
            output_beta.append(beta_h)
            output_gamma.append(gamma_h)
            c_t_minus_1_beta = c_t_beta
            c_t_minus_1_gamma = c_t_gamma
            h_t_minus_1_beta = h_t_beta
            h_t_minus_1_gamma = h_t_gamma
        

        # calculate decoder output
        # as input, takes hidden values of final layer from all input tokens (flattened)
        # LSTM eq. 7
        w_d = weights["decoder.weight"]
        bias_z = weights["decoder.bias"]

        output_beta = torch.stack(output_beta)
        output_gamma = torch.stack(output_gamma)
        flattened_beta = output_beta.view(output_beta.shape[0] * output_beta.shape[1], output_beta.shape[2])
        flattened_gamma = output_gamma.view(output_gamma.shape[0] * output_gamma.shape[1], output_gamma.shape[2])

        beta_z = flattened_beta @ torch.t(w_d)
        gamma_z = flattened_gamma @ torch.t(w_d)
        beta_z = beta_z.view(output_beta.shape[0], output_beta.shape[1], beta_z.shape[1])
        gamma_z = gamma_z.view(output_gamma.shape[0], output_gamma.shape[1], gamma_z.shape[1])

        return beta_z, gamma_z, bias_z
        

def decomp_activation_three(a, b, c, activation):
    """
    Linearize nonlinear activation function with three inputs using Shapely approximation
    """
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    c_contrib = activation(c)

    full = a_contrib + b_contrib + c_contrib
    assert torch.allclose(full, activation(a + b + c), 1e-3, 1e-3)

    return a_contrib, b_contrib, c_contrib


def decomp_activation_two(a, b, activation):
    """
    Linearize nonlinear activation function with two inputs using Shapely approximation
    """
    zero = torch.zeros(a.shape)

    a_contrib = 0.5 * (activation(a + b) - activation(b) + activation(a) - activation(zero))
    b_contrib = 0.5 * (activation(a + b) - activation(a) + activation(b) - activation(zero))

    full = a_contrib + b_contrib
    assert torch.allclose(full, activation(a + b), 1e-3, 1e-3)

    return a_contrib, b_contrib