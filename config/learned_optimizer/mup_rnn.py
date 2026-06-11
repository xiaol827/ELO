

learned_optimizer_args = dict(
    class_="MuRNNMLPLOpt",  
    kwargs=dict(
        step_multiplier = 0.01,
        magnitude_rate = 0.001,
        hidden_size = 32,
        hidden_layer = 2,
        from_mlp_size = 16,
        from_lstm_size = 18,
        lstm_to_ff = 17,
        lstm_hidden_size = 64,
        decays = (0.5, 0.9, 0.99, 0.999, 0.9999),
        zero_lstm_features = False,
        mup_to_lstm = True,
      ))


      