import chemprop
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
if __name__ == '__main__':
    def charge_pre(smiles):

            arguments = [
            '--test_path', '/dev/null',
             '--preds_path', '/dev/null',
            #'--constraints_path', 'atomic_bond_constraints.csv',
            '--checkpoint_dir', 'CM5_charge/charge_checkpoints'
            ]

            args = chemprop.args.PredictArgs().parse_args(arguments)

            #model_objects = chemprop.train.load_model(args=args)
            chg_preds = chemprop.train.make_predictions(args=args,smiles=smiles)
            return chg_preds
    def other_pre(smiles):
            arguments = [
                '--test_path', '/dev/null',
                '--preds_path', '/dev/null',
                '--checkpoint_dir', 'qm9/qm9_checkpoints'
            ]

            args = chemprop.args.PredictArgs().parse_args(arguments)
            other_preds = chemprop.train.make_predictions(args=args, smiles=smiles)
            return other_preds



    def reduced(list1):
        return list(np.array(list1).flatten())
    #pre_data=pd.read_csv(r'/home/lafremd5/data/Luohaoran/DN_ML/DN/qm9/qm9.csv')
if __name__ == '__main__':
    smiles=[['CS(=O)C']]
    props=[]
    charges=charge_pre(smiles)
    others = other_pre(smiles)
    for charge,other in zip(charges,others):
        charge= reduced(reduced(charge))
        qmin=[(min(charge)+0.78266)/0.57291]
        qmax=[(max(charge)-0.100328)/0.65799]
        other[2]=(other[2]+0.361)/0.169
        other[3]=(other[3]+0.247)/0.338
        other[0]=(other[0]-0.000007)/(7.13211-0.000007)
        other[1]=(other[1]-4.551131)/198.008069
        other[0],other[1],other[2],other[3]=other[2],other[3],other[0],other[1]
        prop=[qmin+qmax+other]
        props.append(prop)
    #props=pd.DataFrame(props)

    rf = joblib.load('rf.pkl')
    dns=[]
    for i in range(0,len(props)):
        dn=rf.predict(props[i])*37
        dns.append(dn)
    data_save=pd.DataFrame({'smiles':smiles,'DN':dns})
    data_save.to_csv('output.csv')

