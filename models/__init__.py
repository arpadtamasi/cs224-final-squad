def create_model(name, word_vectors, hidden_size, drop_prob=0.):
    if name == 'baseline':
        from models.bidaf import BiDAF
        return BiDAF(word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob)
