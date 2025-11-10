def main():
    import torch
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    attn_scores = inputs @ inputs.T

    print(f"scores: {attn_scores}\n")
    
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

    print(f"weigths: {attn_weights}\n")

    context_vecs = attn_weights @ inputs

    print("contexts:", context_vecs)
            

if __name__ == "__main__":
   main()
