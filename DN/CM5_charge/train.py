import chemprop
import pandas as pd
if __name__ == '__main__':
    arguments = [
    '--data_path', 'charge.csv',
    '--smiles_columns', 'smiles',
    #'--target_columns','charge',
    '--is_atom_bond_targets',
    '--adding_h',
    '--dataset_type', 'regression',
    '--extra_metrics','mae','r2',
    '--save_dir', 'charge_chekpoints'
    # '--data_path', 'charge.csv',
    # '--dataset_type', 'regression',
    # '--checkpoint_dir', 'charge_checkpoints',
    # '--save_dir', 'charge_chekpoints_2'
    ]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mae , r2 = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
