import pickle
import argparse
import time
import json
import random
import os
from Models import *
from Utils import *


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset: yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--method', type=str, default='IEM_pos', help='how to process data: normal/IEM/IEM_pos')
parser.add_argument('--hidden_size', type=int, default=200 , help='hidden state size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--is_train_eval', type=bool, default=False, help='eval train to prevent from over-fitting ')
parser.add_argument('--max_len', type=int, default=10, help='sequence max length')
parser.add_argument('--cl_lambda', type=float, default=10, help='cl weight')
opt = parser.parse_args()

n_node = 37484


def main():
    """  ---------load model--------------- """
    model = trans_to_cuda(SrIEMPosCL(opt, n_node))

    # preliminaries
    seed_torch(2020)
    log_filename = "./results/log_" + model.__class__.__name__ + ".txt"
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename=log_filename, filemode='w')
    m_print(json.dumps(opt.__dict__, indent=4))
    start = time.time()

    ''' ---------process data------------- '''
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, method=opt.method, shuffle=True, max_len=opt.max_len)
    test_data = Data(test_data, method=opt.method, shuffle=False, max_len=opt.max_len)

    # unlike the last version, this one encapsulate train and test
    hit, mrr = train_and_test(model, train_data, test_data, opt)
    m_print("best result: \tRecall@20:\t{}\tMMR@20:\t{}".format(hit, mrr))
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == "__main__":
    main()
