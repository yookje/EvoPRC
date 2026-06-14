##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM =1
##########################################################################################
# Path Config

import os
import sys
import numpy as np
import constant as const
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.Test_All.TSPTester import TSPTester as Tester

##########################################################################################
# parameters


b = os.path.abspath(".").replace('\\', '/')
b2 = os.path.abspath("..").replace('\\', '/')

env_params = {
    # 'problem_size': 100,
    'pomo_size': 1,
    'k_nearest': 1,
    'beam_width': 16,
    'decode_method': 'greedy',
    'mode': 'test',
    'test_in_tsplib': False,
    'tsplib_path': None,
    'data_path': None,
    'load_way': 'allin',
    'sub_path': False,
    'budget': 10,
    'PRC': True,
    'repair_max_sub_length': 1000,
    'random_insertion': True #False
}

model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'use_k_nearest': True,
    'k_nearest_num':1000
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': None,  # directory path of pre-trained model and log files saved.
    },
    'test_episodes': 16,  # 65
    'test_batch_size': 4,
}

logger_params = {
    'log_file': {
        'desc': 'test__uniform1k_greedy',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main_test(path, prob_size, PRC, repair_max_sub_length, budget, random_insertion):
    logger_params['log_file']['desc']=f'test_{prob_size}_RandomInsertion_{random_insertion}_PRC_{PRC}'

    create_logger(**logger_params)
    tester_params['model_load'] = {
        'path': path,
    }

    tester_params['test_episodes'] = datas[prob_size][1]
    tester_params['test_batch_size'] = datas[prob_size][2]
    env_params['data_path'] = b2 + datas[prob_size][0]
    env_params['PRC'] = PRC
    env_params['repair_max_sub_length'] = repair_max_sub_length
    env_params['budget'] = budget
    env_params['random_insertion'] = random_insertion

    if prob_size==-1:
        env_params['test_in_tsplib'] = True
        env_params['tsplib_path'] = b2 + datas[prob_size][0]
        env_params['data_path'] = b2 + datas[prob_size][0]


    if not random_insertion and budget==0: # (purely greedy search)
        model_params['use_k_nearest'] = False

    _print_config()
    #tester = Tester(env_params=env_params,model_params=model_params,tester_params=tester_params)
    #save for random insertion
    tester = Tester(env_params=env_params,model_params=model_params,tester_params=tester_params, prob_size=prob_size)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap



def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":

    datas = {
        #1000: ['/data/output50.txt', 3, 3*const.POP_SIZE] #path, test_num, batch_size *batch_size=test_num * population_size]
        #1000: ['/data/tmp.txt', 1, 1*const.POP_SIZE], #path, test_num, batch_size *batch_size=test_num * population_size]
        #1000: ['/data/MCTS_tsp1000_test_concorde.txt',  3, 3*const.POP_SIZE],
        #1000: ['/data/MCTS_tsp1000_test_concorde.txt', 128, 128],
        1000: ['/data/MCTS_tsp1000_test_concorde.txt', 128, 32*const.POP_SIZE],
        #5000: ['/data/test_tsp5000_lkh3_n16.txt', 4, 4],
        5000: ['/data/test_tsp5000_lkh3_n16.txt', 16, 4*const.POP_SIZE],
        10000: ['/data/validation_TSP10000_n16.txt', 16, 4*const.POP_SIZE],

        #50000:  ['/data/test_tsp50000_lkh3_n16.txt', 16, 2*const.POP_SIZE],#*const.POP_SIZE], #['/data/test_tsp50000_lkh3_n16.txt', 16, 8],
        50000:  ['/data/test_tsp50000_lkh3_n16.txt', 16, 2*const.POP_SIZE],
        #100000: ['/data/test_tsp100000_lkh3_n16.txt', 16, 4], 
        100000: ['/data/test_tsp100000_lkh3_n16.txt', 16, 4], 
        -1:["/data/TSPlib_scale_ge_1K_n33_ascending.txt", 33, 1, 0]
    }

    model_path = {
        #1000 : './result/checkpoint-tsp1k.pt'
        1000:   './result/checkpoint-tsp1k.pt',
        5000:   './result/checkpoint-tsp5k.pt',
        10000:  './result/checkpoint-tsp10k.pt',
        50000:  './result/checkpoint-tsp50k.pt',
        100000: './result/checkpoint-tsp100k.pt',
        -1:     './result/checkpoint-tsp1k.pt',
    }

    PRC = True
    repair_max_sub_length = 1000

    # purely greedy search: random_insertion=False, budget=0
    # Initilize the solution by random insertion and refine it by PRC: random_insertion=True, budget>0
    budget = 0 #10
    random_insertion = False #True #choose beam, sampling or greedy
    """
    trials = 5
    trial_results = [] 


    for prob_size in [5000] : #, 5000, 10000, 50000, 100000, -1]: #문제에 따른 수정필요
        for t in range(trials):
            score_optimal, score_student, gap = main_test(
                                                        model_path[prob_size],
                                                        prob_size, PRC, repair_max_sub_length,
                                                        budget, random_insertion)
            trial_results.append(gap)
    
    print(f"\n finished {trials} trials min {np.min(trial_results)}  max {np.max(trial_results)} avg : {np.mean(trial_results)} std {np.std(trial_results)}")

    """
    for prob_size in [5000]:#[  1000, 5000, 10000, 50000, 100000]  : #, -1]: #문제에 따른 수정필요
        score_optimal, score_student, gap = main_test(
                                                      model_path[prob_size],
                                                      prob_size, PRC, repair_max_sub_length,
                                                      budget, random_insertion)
    """
