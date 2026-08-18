"""
Train SAL reconstruction on a single object.
Usage: python train_sal_object.py <object_name> [--epochs 2000]

Example: python train_sal_object.py bottle_shampoo --epochs 1000
"""
import os
import sys
import argparse

def train_object(object_name, epochs=2000, batch_size=1):
    """Train SAL on a single object"""
    
    sal_code_dir = "third_party/SCUTSurface/reconstruction/SAL/code"
    
    # Check if object exists
    scan_path = f"third_party/SCUTSurface/reconstruction/SAL/data/real_objects/points/{object_name}/{object_name}_scan.xyz"
    gt_path = f"third_party/SCUTSurface/reconstruction/SAL/data/real_objects/points_iou/{object_name}/{object_name}_gt.xyz"
    
    if not os.path.exists(scan_path):
        print(f"Error: Scan file not found: {scan_path}")
        print(f"Available objects:")
        points_dir = "third_party/SCUTSurface/reconstruction/SAL/data/real_objects/points"
        if os.path.exists(points_dir):
            for obj in os.listdir(points_dir):
                print(f"  - {obj}")
        sys.exit(1)
    
    print(f"Training SAL on: {object_name}")
    print(f"  Scan: {scan_path}")
    print(f"  GT:   {gt_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Update config with object path
    config_template = f"""train{{
    plot_frequency = 100
    preprocess = True
    auto_decoder = False
    latent_size = 0
    expname = real_objects_{object_name}
    dataset_path = ../data/real_objects/points/{object_name}/{object_name}_scan.xyz
    adjust_lr = False
    dataset = datasets.recon_dataset.ReconDataSet
    data_split = none

    learning_rate_schedule = [{{ "Type" : "Step",
                              "Initial" : 0.0005,
                               "Interval" : 500,
                                "Factor" : 0.5
                            }},
                            {{
                                "Type" : "Step",
                                "Initial" : 0.001,
                                "Interval" : 500,
                                "Factor" : 0.5
                            }}]
    network_class = model.network.SALNetwork
}}

plot{{
    resolution = 128
    mc_value = 0.0
    is_uniform_grid = True
    verbose = False
    save_html = True
    save_ply = True
    overwrite = True
}}

network{{
    decode_mnfld_pnts = False
    encoder{{

    }}
    decoder
    {{
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
        dropout = []
        dropout_prob =  0.2
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        latent_in = []
        xyz_in_all = False
        activation = None

        latent_dropout = False
        weight_norm = True
    }}

    loss{{
        loss_type = model.loss.SALLoss
        properties{{
            manifold_pnts_weight = 0
            unsigned = True
        }}
    }}
}}
"""
    
    # Write config
    config_path = f"{sal_code_dir}/confs/{object_name}.conf"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_template)
    
    print(f"Created config: {config_path}")
    print()
    print("Starting training...")
    print(f"Command: cd {sal_code_dir} && python training/exp_runner.py --batch_size {batch_size} --nepoch {epochs} --conf confs/{object_name}.conf --workers 1")
    print()
    print("Note: Training will create outputs in:")
    print(f"  - Checkpoints: {sal_code_dir}/exps/real_objects_{object_name}/checkpoints/")
    print(f"  - Plots: {sal_code_dir}/exps/real_objects_{object_name}/plots/")
    print()
    print("To evaluate and generate mesh after training:")
    print(f"  cd {sal_code_dir}")
    print(f"  python evaluate/evaluate.py --conf confs/{object_name}.conf --checkpoint <epoch> --split none")
    print()
    
    # Actually run training
    cmd = f"cd {sal_code_dir} && python training/exp_runner.py --batch_size {batch_size} --nepoch {epochs} --conf confs/{object_name}.conf --workers 1"
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAL on a single object')
    parser.add_argument('object_name', help='Name of the object to train on')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    train_object(args.object_name, args.epochs, args.batch_size)
