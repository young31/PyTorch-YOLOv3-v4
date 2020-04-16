import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", '-n', type=int, default=1, help="number of classes")
    opt = parser.parse_args()

    os.chdir('./data/custom')
    os.system('python custom.py')
    os.chdir('../../config')
    os.system('custom-cfg.py -n %d'%opt.n_classes)
    print('finished')