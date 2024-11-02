import chemprop
if __name__ == '__main__':
    arguments = [
    '--data_path', 'qm9.csv',
    '--smiles_columns', 'smiles',
    '--target_columns','mu','alpha','homo','lumo',
    #'--atom_targets',['charge'],
    #'--is_atom_bond_targets',
    '--dataset_type', 'regression',
    '--save_dir', 'qm9_chekpoints_2'
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mae , r2 = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)