from decomposition import lstm_forward, contextual_decomposition

def get_gender_preference(model, inputs, start, end, intercepts=False, decoder_bias=True):
    """
    Get gender contribution towards predicting "he" over "she" for given focus span [start, end).
    """
    if intercepts:
        beta, gamma, bias = contextual_decomposition(model, inputs, 0, 0)
    else:
        beta, gamma, bias = contextual_decomposition(model, inputs, start, end)

    beta = beta[-1, :].squeeze()
    gamma = gamma[-1, :].squeeze()
    z = beta + gamma + bias

    if decoder_bias:
        beta += bias

    # index of "she" and "he" in vocab file
    relative_she = beta[6]/z[6]
    relative_he = beta[35]/z[35]
    return relative_he - relative_she


def get_bias(model, inputs, s1, s2, o1, o2):
    """
    Returns gender contribution towards predicting "he" over "she" for subject only, phrase between subject and object, object only, and phrase between object and pronoun, in that order.

    inputs: input sentence tokens

    s1: subject start index
    s2: subject end index

    o1: object start index
    o2: subject end index
    """
    subj_pref = get_gender_preference(model, inputs, s1, s2 + 1)
    inter1_pref = get_gender_preference(model, inputs, s2 + 1, o1)
    obj_pref = get_gender_preference(model, inputs, o1, o2 + 1)
    inter2_pref = get_gender_preference(model, inputs, o2 + 1, len(inputs))

    return subj_pref, inter1_pref, obj_pref, inter2_pref    
    
    
def predict_gender(model, inputs, s1, s2, o1, o2):
    """
    Returns fraction of sentences predicted "he" over "she". Tensor contains scores for full model, subject only (no bias), subject only (with bias), object only (no bias), object only (with bias), intercept only (no bias), and intercept only (with bias), in that order.

    inputs: input sentence tokens

    s1: subject start index
    s2: subject end index

    o1: object start index
    o2: subject end index
    """
    full = lstm_forward(model, inputs)

    subj, _, bias = contextual_decomposition(model, inputs, s1, s2 + 1)
    subj_bias = subj + bias

    obj, _, bias = contextual_decomposition(model, inputs, o1, o2 + 1)
    obj_bias = obj + bias

    intercept, _, bias = contextual_decomposition(model, inputs, 0, 0)
    intercept_bias = intercept + bias

    predictions = []
    for logits in [full, subj, subj_bias, obj, obj_bias, intercept, intercept_bias]:
        logits = logits[-1, :].squeeze()
        # index of "she" and "he" in vocab file
        logit_she = logits[6]
        logit_he = logits[35]
        if logit_he >= logit_she:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions