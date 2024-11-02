import chemprop

arguments = [
    '--test_path', '/dev/null',
    '--preds_path', 'output',
    '--checkpoint_dir', 'qm9_checkpoints'
]

args = chemprop.args.PredictArgs().parse_args(arguments)

#model_objects = chemprop.train.load_model(args=args)

smiles = [['CCC'], ['CCCC'], ['OCC']]
preds = chemprop.train.make_predictions(args=args, smiles=smiles)