
import torch

import os
from logging import getLogger

from TSP.Test_All.TSPEnv import TSPEnv as Env
from TSP.Test_All.TSPModel import TSPModel as Model

import time


from utils.utils import *
import random

import constant as const


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,prob_size):
        


        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)

        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))

        ## variables for ga ########################################

        #save random insertion
        #self.random_insertion_list = []
        
        """
        if prob_size == 1000:
            pt_path = '/workspace/SIL/TSP/Test_All/1k_random_insertion.pt'
        elif prob_size == 5000:
            pt_path = '/workspace/SIL/TSP/Test_All/5k_random_insertion.pt'
        elif prob_size == 10000:
            pt_path = '/workspace/SIL/TSP/Test_All/10k_random_insertion.pt'
        elif prob_size == 50000:
            pt_path = '/workspace/SIL/TSP/Test_All/50k_random_insertion.pt'
        elif prob_size == 100000:
            pt_path = '/workspace/SIL/TSP/Test_All/100k_random_insertion.pt'

        self.counter =0
        self.loaded_random_insertion_list = torch.load(pt_path)

        self.loaded_random_insertion_tensor = torch.cat(self.loaded_random_insertion_list, dim=0)  # [-1, N]
        self.insertion_ptr = 0
        """

    
        ##########################################


    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'] )

        k_nearest = self.env_params['k_nearest']

        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_le_5000 = []
        problems_gt_5000 = []


        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score,problems_size = self._test_one_batch(episode,batch_size,k_nearest,decode_method,clock=self.time_estimator_2)

            #for save random insertion
            #self.random_insertion_list.append(self.random_insertion )

            print('max_memory_allocated',torch.cuda.max_memory_allocated(device=self.device ) / 1024 / 1024,'MB')

            if self.env.test_in_tsplib:
                if problems_size <= 1000:
                    problems_le_5000.append((score_student_mean - score) / score)
                elif 5000 < problems_size:
                    problems_gt_5000.append((score_student_mean - score) / score)

                print('problems_le_5000 mean gap:', np.mean(problems_le_5000), len(problems_le_5000))
                print('problems_gt_5000 mean gap:', np.mean(problems_gt_5000), len(problems_gt_5000))

        
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            if all_done and not self.env.test_in_tsplib:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))

                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100


        #for save random insertion
        #torch.save(self.random_insertion_list, "1k_random_insertion.pt")

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                            before_complete_solution, before_repair_sub_solution,
                                          after_repair_sub_solution,before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size  = int(double_solution.shape[1]/2)


        other_part_1 = double_solution[:,:first_node_index]
        other_part_2 = double_solution[:,first_node_index+length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index : first_node_index+length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:,None].repeat(1,jjj.shape[1])

        kkk_2 = jjj[index,after_repair_sub_solution]

        if_repair = before_reward>after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]



        return after_repair_complete_solution

        



###########################################################
#
#
#modified for GA
#
#
###############################################################


    def get_next_insertion(self, loaded_b):
        total = self.loaded_random_insertion_tensor.shape[0]
        
        start = self.insertion_ptr
        end = min(start + loaded_b, total)  # 마지막엔 남은 것 전부
        
        loaded_batch = self.loaded_random_insertion_tensor[start:end]  # [b, N] 또는 [남은개수, N]
        
        self.insertion_ptr = end
        
        return loaded_batch

    def initialize_population_random_insertion(self, problems, pop_size, batch_size, problem_size):
        from utils.insertion import random_insertion
        """
        Random Insertion으로 초기 population 생성
        
        Args:
            problems: (batch_size, problem_size, 2) - TSP 좌표 (tensor)
            pop_size: int - population 크기 (= width)
        
        Returns:
            orders: (batch_size*pop_size, problem_size) - 초기 해 집단
        """
        
        #"""
        orders = [torch.randperm(problem_size) for _ in range(pop_size)]
        
        # 기존 SIL 코드와 순서가 다름 
        pi_all = [
            random_insertion(instance, orders[order_id])[0]
            for instance in problems            # batch_size번
            for order_id in range(pop_size)        # pop_size번
            
        ]
        
        pi_all = np.array(pi_all, dtype=np.int64)
        # shape: (batch_size* pop_size ,  problem_size)
        
        # Tensor 변환
        result = torch.tensor(pi_all, dtype=torch.long)
        
        #"""

        #save for random insertion
        #result=self.loaded_random_insertion_list[self.counter]
        return result
    

    def amplified_solution(self,solution, B, N, K):
       
        # 1. [B, N] → [B*K, N] 직접 repeat (expand+clone 대신)
        expanded_flat = solution.repeat_interleave(K, dim=0)  # [B*K, N]
        
        # 2. inverse + roll 마스크 생성
        inverse_mask = torch.rand(B * K, device=self.device) < 0.5  # [B*K]
        roll_shifts = torch.randint(0, N, (B * K,), device=self.device)
        


        # 3. inverse 적용: flip된 index 생성
        idx = torch.arange(N, device=self.device)  # [N]
        # inverse=True면 (N-1-idx), False면 idx
        base_idx = torch.where(
            inverse_mask.unsqueeze(1),
            N - 1 - idx,
            idx.unsqueeze(0).expand(B * K, -1)
        )  # [B*K, N]
        
        # 4. roll 적용: index shift
        final_idx = (base_idx - roll_shifts.unsqueeze(1)) % N  # [B*K, N]
        
        # 5. 한번의 gather로 완료
        result = torch.gather(expanded_flat, dim=1, index=final_idx)
        
        return result


    def run_prc(self, given_node_list, batch_size, decode_method,  clock=None, omega = None, budget=0, needed_prc = 0):

        best_select_node_list = given_node_list

        #print(f"\n \t\t ******** origin_problem {self.origin_problem. size()}  best_se {best_select_node_list.size()}")
        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,test_in_tsplib=self.env.test_in_tsplib)

        max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024


        escape_time, _ = clock.get_est_string(1, 1)

        gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

        if self.env.test_in_tsplib:
            self.logger.info("greedy, name:{}, gap:{:6f} %,  Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                self.name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
        else:
            self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                gap, escape_time,  current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after ))

        #budget = self.env_params['budget']
        #print(f"\n budget {budget}")


        origin_problem_size = self.origin_problem.shape[1]

        # torch.save(best_select_node_list, f'TSP{origin_problem_size}_step0.pt')

        origin_batch_size = batch_size

        #adjust omega
        repair_max_sub_length = self.env_params['repair_max_sub_length']

        lower_bound = 4

        repair_max_sub_length = min(origin_problem_size, repair_max_sub_length)
       
        if needed_prc == 0 :
            length_all = torch.randint(low=lower_bound, high=repair_max_sub_length + 1, size=[budget])
            length_all, sorted_idx = torch.sort(length_all, descending=True)

            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])

            #print(f"\n low : {lower_bound}  max_sub_length : {repair_max_sub_length} ")
            
           

        else: #offspring을 위한...
          
            length_all = torch.randint(low=lower_bound, high=repair_max_sub_length + 1, size=[budget])           
            length_all, sorted_idx = torch.sort(length_all, descending=True)

            #first_index_all의 처음은 xover 후 손상된 해를 수정하기 위해  수정
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget], device=self.device)
            first_index_all[0] = origin_problem_size-needed_prc-1

            #print(f"\n low : {lower_bound}  max_sub_length : {repair_max_sub_length} ")

       

        #print(f"\n\n check length_all {length_all}")
        



        for bbbb in range(budget):

            self.env.problems = self.origin_problem.clone().detach()

            if bbbb != 0 :
                best_select_node_list = self.env.random_inverse_solution(best_select_node_list)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                    best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
            

            

            #for check
            self.prc_start_time = time.time()
            #print(f"\n------------- \n current_best_length {current_best_length} initialized {best_select_node_list}  ")


            if_PRC = self.env_params['PRC']

            if if_PRC:
                partial_solution_length, first_node_index, length_of_subpath, double_solution, \
                origin_sub_solution, index4, factor = \
                    self.env.destroy_solution_PRC(self.env.problems, best_select_node_list, length_all[bbbb],
                                                    first_index_all[bbbb], )
                #print(f"\n\tsubpath length {length_of_subpath}")
            else:
                partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list)
            

            before_reward = partial_solution_length

            before_repair_sub_solution = self.env.solution

            self.env.batch_size = before_repair_sub_solution.shape[0]

            current_step = 0

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            state, reward, reward_student, done = self.env.pre_step()

            while not done:
                if current_step == 0:
                    selected_teacher = self.env.solution[:, -1]
                    selected_student = self.env.solution[:, -1]

                elif current_step == 1:
                    selected_teacher = self.env.solution[:, 0]
                    selected_student = self.env.solution[:, 0]

                else:
                    selected_teacher, _,_,selected_student = self.model(
                        state,self.env.selected_node_list,self.env.solution,current_step,
                        decode_method=decode_method,repair = True)

                current_step += 1
                state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

            ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)

            after_reward = reward_student

            if if_PRC:
                after_repair_complete_solution = self.env.decide_whether_to_repair_solution_PRC(
                    ahter_repair_sub_solution, before_reward, after_reward, double_solution,
                    origin_batch_size, origin_sub_solution, index4, factor)
            else:
                after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                    ahter_repair_sub_solution, before_reward, after_reward,
                    first_node_index, length_of_subpath, double_solution)

            
            best_select_node_list = after_repair_complete_solution
            
        
            
            self.prc_time = (time.time() - self.prc_start_time)
            #print("******* prc time ={:.03f}".format(self.prc_time))
            


            current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                    best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
            

            
            escape_time,_ = clock.get_est_string(1, 1)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            
            #for check solution
            #print(f"\n\n in prc self.optimal_length {self.optimal_length}")
            #print(f"\n\t tour length : {current_best_length} gap {gap} \ntour {best_select_node_list.size()} {best_select_node_list}")
            
            if self.env.test_in_tsplib:

                self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                        bbbb, self.name, gap, escape_time, current_best_length.mean().item(),
                        self.optimal_length.mean().item()))
            else:
                self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                    bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after))


        #end PRC
        return best_select_node_list, current_best_length


    def _test_one_batch(self, episode, batch_size,k_nearest,decode_method,clock=None): #_ga

        self.model.eval()


        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

        print('max_memory_allocated before', max_memory_allocated_before, 'MB')

        torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():

            self.env.load_problems(episode, batch_size,only_test=True)
            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                       test_in_tsplib=self.env.test_in_tsplib,
                                                                       need_optimal=True)
                self.optimal_length = optimal_length[episode]
                self.name = name[episode]

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            IF_random_insertion = self.env_params['random_insertion']

            

            if IF_random_insertion:
                from utils.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                #modified 
                batch_size, problem_size, _ = dataset.shape


                #"""
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)
                """
                best_select_node_list = self.get_next_insertion(batch_size) 
                best_select_node_list = best_select_node_list.to(self.device)
                
                
                print(f"\n best_select_node_list before expaned {best_select_node_list.size()}")
                """
                #modified for ga
                best_select_node_list = self.amplified_solution(best_select_node_list, batch_size, problem_size, const.POP_SIZE)
                #print(f"\n best_select_node_list  {best_select_node_list.size()}") #[batch_size, pop_size, N]
                #print("\n******* random insertion time ={:.03f}".format(self.ri_time))

            else:

                B_V = batch_size * 1
                current_step = 0
                state, reward, reward_student, done = self.env.pre_step()
                from tqdm import tqdm
                with tqdm(total=self.env.problem_size) as pbar:
                    while not done:
                        pbar.update(1)

                        if current_step == 0:
                            selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                            selected_student = selected_teacher

                        else:
                            selected_teacher, _,_,selected_student = self.model(
                                state,self.env.selected_node_list,self.env.solution,current_step,
                                decode_method=decode_method)

                        current_step += 1

                        state, reward,reward_student, done = self.env.step(selected_teacher, selected_student)

                    
                best_select_node_list = self.env.selected_node_list


            

            ### population inizialization
            from ga_component import Population


        
            #Population 생성
            self.population = Population(best_select_node_list, batch_size , const.POP_SIZE, problem_size, num_parents=const.NUM_PARENTS,selection_pressure=const.SELECTION_PRESSURE) 

            
            #doing PRC          
            #PRC를 위해서 차원맞추기
            coord = self.origin_problem # [B, N, 2]
            self.origin_problem = self.origin_problem.repeat_interleave(const.POP_SIZE, dim=0) 
            new_batch_size = batch_size * const.POP_SIZE  # num_of_instances(==batch) * population_size
            updated_solution, current_best_length =self.run_prc(best_select_node_list, new_batch_size, decode_method, clock,omega = None, budget = const.FIRST_BUDGET) #omega = int(problem_size*0.6) 
            
       

            #population의 해 품질 평가
            self.population.eval_population(updated_solution, current_best_length)


            #variable for plotting
            plot_var_generation = list(range(0, const.MAX_GENERATION + 1))
            plot_var_best_gap = []
            plot_var_pop_gap = []

            pops_length=self.population.get_information() #[B*k]
            pops_length = pops_length.view(-1, const.POP_SIZE).mean(dim=1)  # [B]
            pop_gap = ((pops_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            plot_var_pop_gap.append(pop_gap)
            #print(f"\n population avg gap  {pop_gap }")

            #population's best 초기화
            current_best_length = self.population.get_best()
            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100       
            plot_var_best_gap.append(gap)
            before_best = gap
            #print(f"\n population's best gap   {before_best }")
       
            
            ### main loop
            for g in range(const.MAX_GENERATION):
                #print(f"\n\n\n\n\n *******this is {g}th generation")
                #selection
                self.population.roulette_wheel_selection()
                #self.population.roulette_wheel_selection_wo_shared_fitness()
                
                #crossover
                offspring, leftover_length = self.population.onepoint_crossover() #[batch_size * 2 , N]

                #local search
                #print(f"\n\n ======local search")

                #자식이 2개일 때...
                #PRC를 위한 차원 맞추기 : origin coord와 pred solution 셋팅 # [B, N, 2]
                #self.origin_problem = coord.repeat_interleave(2, dim=0) #PRC를 위해서 차원맞추기 [B*2, N, 2]
                #new_batch_size = batch_size * 2  # num_of_instances(==batch) * 2(#offsprings)

                self.origin_problem = coord # [B, N, 2]
                
      
                updated_off_sol, update_off_length =self.run_prc(offspring, batch_size, decode_method, clock, omega = None, budget = const.SECOND_BUDGET, needed_prc = leftover_length) #xover 후 해의 품질을 높이기위한 prc
        
                #updated_off_sol, update_off_length =self.run_prc(offspring, batch_size, decode_method, clock, omega = None , budget = const.SECOND_BUDGET) #omega = int(problem_size)//2
                
                #replace
                self.population.replace(updated_off_sol, update_off_length )

                #mutation
                #self.population.mutation()
                
                """
                print(f"\n\n \t\t *******just fisnihed {g}th generation \n\n\n\n\n")
                pops_length=self.population.get_information() #[B*k]
                pops_length = pops_length.view(-1, const.POP_SIZE).mean(dim=1)  # [B]
                pop_gap = ((pops_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                plot_var_pop_gap.append(pop_gap)
                print(f"\n population avg gap  {pop_gap }")

                current_best_length = self.population.get_best()
                gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                change = before_best - gap
                
               
                if (change > 0.0) :
                    print(f"\n best gap changed!  change amount : {change}  best gap :  {gap }")
                else:
                    print(f"\n not changed at all!  best gap so far :  {gap }")
                plot_var_best_gap.append(gap)
                before_best = gap
                """
    
           
            
 

        current_best_length = self.population.get_best()
        gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
    
        return self.optimal_length.mean().item(),current_best_length.mean().item(), current_best_length.mean().item(),self.env.problem_size


####################################
#
# multi start
#
#######################################

    def run_prc_multistart(self, given_node_list, batch_size, decode_method,  clock=None, omega = None, budget=0, needed_prc = 0):

        best_select_node_list = given_node_list

        #print(f"\n \t\t ******** origin_problem {self.origin_problem. size()}  best_se {best_select_node_list.size()}")
        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,test_in_tsplib=self.env.test_in_tsplib)

        max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024


        escape_time, _ = clock.get_est_string(1, 1)

        gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

        if self.env.test_in_tsplib:
            self.logger.info("greedy, name:{}, gap:{:6f} %,  Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                self.name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
        else:
            self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                gap, escape_time,  current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after ))

        #budget = self.env_params['budget']
        print(f"\n budget {budget}")


        origin_problem_size = self.origin_problem.shape[1]

        # torch.save(best_select_node_list, f'TSP{origin_problem_size}_step0.pt')

        origin_batch_size = batch_size

        #adjust omega
        repair_max_sub_length = self.env_params['repair_max_sub_length']

        
        lowerbound = 4
        

        if needed_prc == 0 :
            length_all = torch.randint(low=lowerbound, high=repair_max_sub_length + 1, size=[budget])
            
            #length_all, sorted_idx = torch.sort(length_all, descending=True)

            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])

            #print(f"\n low : {lowerbound}  max_sub_length : {repair_max_sub_length} ")
            
           

        else: #offspring을 위한...
          
            length_all = torch.randint(low=lowerbound , high=repair_max_sub_length + 1, size=[budget])           
            
            length_all, sorted_idx = torch.sort(length_all, descending=True)
        
            #first_index_all의 처음은 xover 후 손상된 해를 수정하기 위해  수정
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget], device=self.device)
            first_index_all[0] = origin_problem_size-needed_prc-1

            #print(f"\n low : {lowerbound }  max_sub_length : {repair_max_sub_length} ")

       

        print(f"\n\n check length_all {length_all}")
        



        for bbbb in range(budget):

            self.env.problems = self.origin_problem.clone().detach()

            if bbbb != 0 :
                best_select_node_list = self.env.random_inverse_solution(best_select_node_list)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                    best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
            

            

            #for check
            self.prc_start_time = time.time()
            #print(f"\n------------- \n current_best_length {current_best_length} initialized {best_select_node_list}  ")


            if_PRC = self.env_params['PRC']

            if if_PRC:
                partial_solution_length, first_node_index, length_of_subpath, double_solution, \
                origin_sub_solution, index4, factor = \
                    self.env.destroy_solution_PRC(self.env.problems, best_select_node_list, length_all[bbbb],
                                                    first_index_all[bbbb], )
                #print(f"\n\tsubpath length {length_of_subpath}")
            else:
                partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list)
            

            before_reward = partial_solution_length

            before_repair_sub_solution = self.env.solution

            self.env.batch_size = before_repair_sub_solution.shape[0]

            current_step = 0

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            state, reward, reward_student, done = self.env.pre_step()

            while not done:
                if current_step == 0:
                    selected_teacher = self.env.solution[:, -1]
                    selected_student = self.env.solution[:, -1]

                elif current_step == 1:
                    selected_teacher = self.env.solution[:, 0]
                    selected_student = self.env.solution[:, 0]

                else:
                    selected_teacher, _,_,selected_student = self.model(
                        state,self.env.selected_node_list,self.env.solution,current_step,
                        decode_method=decode_method,repair = True)

                current_step += 1
                state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

            ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)

            after_reward = reward_student

            if if_PRC:
                after_repair_complete_solution = self.env.decide_whether_to_repair_solution_PRC(
                    ahter_repair_sub_solution, before_reward, after_reward, double_solution,
                    origin_batch_size, origin_sub_solution, index4, factor)
            else:
                after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                    ahter_repair_sub_solution, before_reward, after_reward,
                    first_node_index, length_of_subpath, double_solution)

            
            best_select_node_list = after_repair_complete_solution
            
        
            
            self.prc_time = (time.time() - self.prc_start_time)
            print("******* prc time ={:.03f}".format(self.prc_time))
            


            current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                    best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
            

            
            escape_time,_ = clock.get_est_string(1, 1)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            
       
            if self.env.test_in_tsplib:

                self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                        bbbb, self.name, gap, escape_time, current_best_length.mean().item(),
                        self.optimal_length.mean().item()))
            else:
                self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                    bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after))


        #end PRC
        return best_select_node_list, current_best_length

    def _test_one_batch_multistart(self, episode, batch_size,k_nearest,decode_method,clock=None): #_multistart

        self.model.eval()


        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

        print('max_memory_allocated before', max_memory_allocated_before, 'MB')

        torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():

            self.env.load_problems(episode, batch_size,only_test=True)
            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                       test_in_tsplib=self.env.test_in_tsplib,
                                                                       need_optimal=True)
                self.optimal_length = optimal_length[episode]
                self.name = name[episode]

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            IF_random_insertion = self.env_params['random_insertion']

            

            if IF_random_insertion:
                from utils.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                #modified 
                batch_size, problem_size, _ = dataset.shape


                """
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)
                """
                best_select_node_list = self.get_next_insertion(batch_size) 
                best_select_node_list = best_select_node_list.to(self.device)
                
                #"""
                #print(f"\n best_select_node_list before expaned {best_select_node_list.size()}")

                #modified for ga
                best_select_node_list = self.amplified_solution(best_select_node_list, batch_size, problem_size, const.POP_SIZE)
                #print(f"\n best_select_node_list  {best_select_node_list.size()}") #[batch_size, pop_size, N]
                #print("\n******* random insertion time ={:.03f}".format(self.ri_time))

            else:

                B_V = batch_size * 1
                current_step = 0
                state, reward, reward_student, done = self.env.pre_step()
                from tqdm import tqdm
                with tqdm(total=self.env.problem_size) as pbar:
                    while not done:
                        pbar.update(1)

                        if current_step == 0:
                            selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                            selected_student = selected_teacher

                        else:
                            selected_teacher, _,_,selected_student = self.model(
                                state,self.env.selected_node_list,self.env.solution,current_step,
                                decode_method=decode_method)

                        current_step += 1

                        state, reward,reward_student, done = self.env.step(selected_teacher, selected_student)

                    
                best_select_node_list = self.env.selected_node_list


            

            ### population inizialization
            from ga_component import Population


           
            #Population 생성
            self.population = Population(best_select_node_list, batch_size , const.POP_SIZE, problem_size, num_parents=const.NUM_PARENTS,selection_pressure=const.SELECTION_PRESSURE) 

            
            #doing PRC          
            #PRC를 위해서 차원맞추기
            coord = self.origin_problem # [B, N, 2]
            self.origin_problem = self.origin_problem.repeat_interleave(const.POP_SIZE, dim=0) 
            new_batch_size = batch_size * const.POP_SIZE  # num_of_instances(==batch) * population_size
            updated_solution, current_best_length =self.run_prc_multistart(best_select_node_list, new_batch_size, decode_method, clock,omega = None, budget = const.FIRST_BUDGET) #omega = int(problem_size*0.6) 
            
       

            #population의 해 품질 평가
            self.population.eval_population(updated_solution, current_best_length)

 
            #population's best 초기화
            current_best_length = self.population.get_best()

            print(f"\n current_best_length {current_best_length.size()}")
       
            
            print("\n done")
            


  
        gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
    
        return self.optimal_length.mean().item(),current_best_length.mean().item(), current_best_length.mean().item(),self.env.problem_size
