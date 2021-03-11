def create_model(
        name,
        hidden_size,
        word_vectors, char_vectors,
        drop_prob=0.1, layer_drop_prob=0.9
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
        dropout_char = 0.05
        freeze_char_embedding = False
        para_limit = 400
        ques_limit = 50

        return QANet(
            word_vectors=word_vectors, char_vectors=char_vectors,
            d_model=d_model, n_head=num_heads,
            len_c=para_limit + 1, len_q=ques_limit + 1,
            dropout=drop_prob, dropout_char=dropout_char,
            freeze_char_embedding=freeze_char_embedding
        )
    elif name == 'claf':
        from models.claf import QANet as ClafQANet

        model_dim = 128
        num_heads = 8
        dropout_char = 0.05
        layer_dropout = layer_drop_prob
        freeze_char_embedding = False
        model_dim = model_dim
        kernel_size_in_embedding = 7
        num_head_in_embedding = 8
        num_embedding_encoder_block = 1
        kernel_size_in_modeling = 5
        num_conv_block_in_modeling = 2
        num_modeling_encoder_block = 7

        return ClafQANet(
            word_vectors=word_vectors, char_vectors=char_vectors,
            aligned_query_embedding=True,freeze_char_embedding=freeze_char_embedding,
            model_dim=model_dim, kernel_size_in_embedding=kernel_size_in_embedding,
            num_head_in_embedding=num_head_in_embedding,num_embedding_encoder_block=num_embedding_encoder_block,
            kernel_size_in_modeling=kernel_size_in_modeling,num_head_in_modeling=num_heads,
            num_conv_block_in_modeling=num_conv_block_in_modeling,
            num_modeling_encoder_block=num_modeling_encoder_block,
            dropout=drop_prob,
            layer_dropout=layer_dropout,
            char_dropout=dropout_char
        )
