model_config = {
    'vae_mid': 20,
    'num_words': 35285,
    'vocab_size': 35285,
    'bow_mid_hid': 512,
    'seq_mid_hid': 512,
    'seq_len': 100,
    'num_heads': 8,
    'dropout': 1,
    'is_traing': True
}

data_config = {
    'data_path': 'data/data.bin',
    'vocabulary_path': 'data/vocabulary.json',
    'stop_words_path': 'data/stop_words'
}

train_config = {
    'batch_size': 2000,
    'epochs': 1000,
    'lr': 3e-3,
    'clip_grad': 20,
    'save_step' : 100,
    'checkpoint_path': 'checkpoint/'
}

eval_config = {
    'batch_size': 1000,
}
