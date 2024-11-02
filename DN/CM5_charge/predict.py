import chemprop
if __name__ == '__main__':
    arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    #'--constraints_path', 'atomic_bond_constraints.csv',
    '--checkpoint_dir', 'charge_checkpoints'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    model_objects = chemprop.train.load_model(args=args)
    smiles = [['CCO']]
    preds = chemprop.train.make_predictions(args=args,smiles=smiles, model_objects=model_objects)
    print(preds)
