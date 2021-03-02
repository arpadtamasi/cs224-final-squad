def create_model(
        name,
        hidden_size,
        word_vectors, char_vectors,
        drop_prob=0.,
):
    if name == 'baseline':
        from models.bidaf import BiDAF
        return BiDAF(
            word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob
        )
    elif name == 'qanet':
        from models.qanet import QANet

        d_model = 96
        num_heads = 8
        drop_prob = 0.
        dropout_char = 0.05
        freeze_char_embedding = False
        para_limit = 400
        ques_limit = 50

        return QANet(
            word_mat=word_vectors, char_mat=char_vectors,
            d_model=d_model, n_head=num_heads,
            len_c=para_limit + 1, len_q=ques_limit + 1,
            dropout=drop_prob, dropout_char=dropout_char,
            freeze_char_embedding=freeze_char_embedding
        )
