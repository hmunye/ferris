def main():
    import torch
    from attention import SelfAttention 
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(789)
    attn = SelfAttention(d_in, d_out)

    print(attn(inputs))


if __name__ == "__main__":
   main()
