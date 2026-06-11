compression_args = dict(
    topk_percent=0.2, # percentage of values to keep
    quantize_bits=4, # number of bits to quantize to
    random_percent=0.1, # percentage of values to randomize
    compression_order = "random,topk,quantize" # order of compression
)