def main():
    import torch
    from attention import MultiHeadAttention
    
    torch.manual_seed(123)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    batch = torch.stack((inputs, inputs), dim=0)

    batch_size, context_length, d_in = batch.shape
    d_out = 2

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)

    print("context vectors:", context_vecs)
    print("context shape:", context_vecs.shape)


if __name__ == "__main__":
   main()
