# ------------------------------------------------------------------
"""
Integration test script for SubMamba training pipeline

"""
# ------------------------------------------------------------------

import torch
from config import read_arguments
from model.Mamba2_encoder import Mamba2 as EncoderMamba2
from model.MLP_head import MLP as RegressionHead
from dataset.dataset import SubwayDelayDataset
from serialization.serialization import get_graph_vertex_order


def test_pipeline():
    print("Initializing configuration...")
    # Load default configuration (train=True)
    config = read_arguments(train=True, print=False, save=False)

    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_id != "-1" else 'cpu')
    print(f"Using device: {device}")

    # 1. Test Dataset Loader
    print("Testing SubwayDelayDataset...")
    dataset = SubwayDelayDataset(
        processed_dir=config.dir_log,  # or your actual processed data dir
        years=config.years_val,
        n_points=config.n_points
    )

    # Create a dummy batch if dataset is empty for testing purposes
    if len(dataset) > 0:
        sample = dataset[0]
        x_dynamic = sample['x_dynamic'].unsqueeze(0).to(device)  # [1, T, P, V]
        x_static = sample['x_static'].unsqueeze(0).to(device)    # [1, P, V_stat]
        graph_vertex_order = sample['graph_vertex_order'].unsqueeze(0).to(device)  # [1, P]
        y_target = sample['y_target'].unsqueeze(0).to(device)    # [1, P, delta_t_f]
    else:
        print("Dataset directory empty. Generating synthetic batch for testing...")
        B, T, P, V_dyn = 2, 4, config.n_points, config.in_dynamic
        V_stat = config.in_static
        x_dynamic = torch.randn((B, T, P, V_dyn), dtype=torch.float32, device=device)
        x_static = torch.randn((B, P, V_stat), dtype=torch.float32, device=device)
        graph_vertex_order = get_graph_vertex_order(method=config.graph_vertex_order, num_nodes=P).to(device).repeat(B, 1)
        y_target = torch.randn((B, P, config.delta_t_f), dtype=torch.float32, device=device)

    print(f"  x_dynamic shape: {x_dynamic.shape}")
    print(f"  x_static shape: {x_static.shape}")
    print(f"  graph_vertex_order shape: {graph_vertex_order.shape}")

    # 2. Test Encoder Backbone
    print("Initializing SubMamba Encoder...")
    encoder = EncoderMamba2(
        in_dynamic=config.in_dynamic,
        in_static=config.in_static,
        embed_dynamic=config.en_embed_dim[0],
        embed_dim=config.en_embed_dim,
        depths=config.en_depths,
        grouping_size=config.en_grouping_size
    ).to(device)

    # 3. Test Regression Head
    print("Initializing Regression Head...")
    head = RegressionHead(
        embed_dim=config.en_embed_dim[-1],
        embed_dim_all=config.en_embed_dim[-1] * config.delta_t_f,
        hidden_dim=config.head_hidden_dim,
        out_dim=config.head_out_dim
    ).to(device)

    # 4. Forward Pass Test
    print("Running forward pass test...")
    encoder.eval()
    head.eval()
    with torch.no_grad():
        # Encoder forward
        encoded_features = encoder(x_dynamic, x_static, graph_vertex_order)
        print(f"  Encoder output shape: {encoded_features.shape}")

        # Simulate multi-step lead-time features for the head
        # x_future list representing delta_t_f forecast steps
        x_future = [encoded_features] * config.delta_t_f

        # Head forward
        predictions = head(encoded_features, x_future)
        print(f"  Regression head output shape: {predictions.shape}")

    print("Pipeline test completed successfully!")


if __name__ == '__main__':
    test_pipeline()