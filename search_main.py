import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
from math import *
from search_utils import *

def get_args():
    parser = argparse.ArgumentParser("Search")
    parser.add_argument('--model', type=str, default='checkpoints', help='dir for loading checkpoint')
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--img-dir', type=str, default='../', help='dir for loading images')
    parser.add_argument('--valid-list', type=str, default='../dataset/valid.txt', help='path to validation list')
    parser.add_argument('--image-size', type=list, default=[192, 256, 320, 384, 448], help='image resolution list')
    
    args = parser.parse_args()
    return args  

# Domination check
def dominate(p, q):
    result = False
    for i, j in zip(p.obj, q.obj):
        if i < j:  # at least less in one dimension
            result = True
        elif i > j:  # not greater in any dimension, return false immediately
            return False
    return result


def non_dominate_sorting(population):
    # find non-dominated sorted
    dominated_set = {}
    dominating_num = {}
    rank = {}
    for p in population:
        dominated_set[p] = []
        dominating_num[p] = 0

    sorted_pop = [[]]
    rank_init = 0
    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominate(p, q):
                dominated_set[p].append(q)
                dominating_num[q] += 1
            elif dominate(q, p):
                dominating_num[p] += 1
                dominated_set[q].append(p)
        # rank 0
        if dominating_num[p] == 0:
            rank[p] = rank_init # rank set to 0
            sorted_pop[0].append(p)

    while len(sorted_pop[rank_init]) > 0:
        current_front = []
        for ppp in sorted_pop[rank_init]:
            for qqq in dominated_set[ppp]:
                dominating_num[qqq] -= 1
                if dominating_num[qqq] == 0:
                    rank[qqq] = rank_init + 1
                    current_front.append(qqq)
        rank_init += 1

        sorted_pop.append(current_front)

    return sorted_pop


class Individual():
    def __init__(self, random_code):
        self.dec = random_code  ## binary
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        
    def __str__(self):
        return self.__dict__


def initialization(pop_size, random_func, eval_func):
    population = []
    for i in range(pop_size):
        random_code = random_func()
        ind = Individual(random_code)
        dice, acc = eval_func(ind.dec)
        ind.obj = [1-dice, 1-acc]
        population.append(ind)

    return population


def evaluation(population, eval_func):
    # Evaluation
    for ind in population:
        dice, acc = eval_func(ind.dec)
        ind.obj = [1-dice, 1-acc]
    
    return population

# one point crossover
def one_point_crossover(p, q):
    gene_length = len(p.dec)
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    k = np.random.randint(gene_length)
    # logging.info("crossover at {}".format(k))
    child1[:k] = p.dec[:k]
    child1[k:] = q.dec[k:]

    child2[:k] = q.dec[:k]
    child2[k:] = p.dec[k:]

    return child1, child2

# Bit wise mutation
def bitwise_mutation(p, p_m, random_func):
    random_code = random_func()
    gene_length = len(random_code)
    p_mutation = p_m # / gene_length
    for i in range(gene_length):
        if np.random.random()<p_mutation:
            # logging.info("Mutated at {} with {}".format(i, p_mutation))
            p.dec[i] = random_code[i]
    return p

# Variation (Crossover & Mutation)
def variation(population, p_crossover, p_mutation, random_func, eval_func):
    offspring = copy.deepcopy(population)
    len_pop = int(np.ceil(len(population) / 2) * 2) 
    candidate_idx = np.random.permutation(len_pop)

    # Crossover
    for i in range(int(len_pop/2)):
        if np.random.random()<=p_crossover:
            individual1 = offspring[candidate_idx[i]]
            individual2 = offspring[candidate_idx[-i-1]]
            [child1, child2] = one_point_crossover(individual1, individual2)
            offspring[candidate_idx[i]].dec[:] = child1
            offspring[candidate_idx[-i-1]].dec[:] = child2

    # Mutation
    for i in range(len_pop):
        individual = offspring[i]
        offspring[i] = bitwise_mutation(individual, p_mutation, random_func)

    # Evaluate offspring
    offspring = evaluation(offspring, eval_func)

    return offspring

# Crowding distance
def crowding_dist(population):
    pop_size = len(population)
    crowding_dis = np.zeros((pop_size, 1))

    obj_dim_size = len(population[0].obj)
    # crowding distance
    for m in range(obj_dim_size):
        obj_current = [x.obj[m] for x in population]
        sorted_idx = np.argsort(obj_current)  # sort current dim with ascending order
        obj_max = np.max(obj_current)
        obj_min = np.min(obj_current)

        # keep boundary point
        crowding_dis[sorted_idx[0]] = np.inf
        crowding_dis[sorted_idx[-1]] = np.inf
        for i in range(1, pop_size - 1):
            crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                      1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                             obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
    return crowding_dis


# Environmental Selection
def environmental_selection(population, n):
    pop_sorted = non_dominate_sorting(population)
    selected = []
    for front in pop_sorted:
        if len(selected) < n:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # select individuals according crowding distance here
                crowding_dst = crowding_dist(front)
                k = n - len(selected)
                dist_idx = np.argsort(crowding_dst, axis=0)[::-1]
                for i in dist_idx[:k]:
                    selected.extend([front[i[0]]])
                break
    return selected

def main():
    args = get_args()
    
    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fh = logging.FileHandler(os.path.join('{}/train-{}{:02}{}'.format(log_dir, local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    # Record file path and content
    logger = logging.getLogger()
    filepath = os.path.abspath(__file__)
    logger.info(filepath)
    logger.info(args)
    
    # with open(filepath, "r") as f:
    #     logger.info(f.read())
    
    ec_save_dir = "ec_save"
    if not os.path.exists(ec_save_dir):
        os.mkdir(ec_save_dir)
        
    dnn = DNN(args.model, args.img_dir, args.valid_list, args.image_size, args.batch_size)
    random_func = dnn.random_code
    eval_func = dnn.eval_solution

    # configuration
    population = []
    pop_size = 40  # Population size
    gen = 100          # Iteration number
    
    n_obj = 2       # Objective variable dimensionality
    p_crossover = 1     # crossover probability
    p_mutation = 0.1      # mutation probability

    # Initialize
    population = initialization(pop_size, random_func, eval_func)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for g in range(gen):

        # Variation
        offspring = variation(population, p_crossover, p_mutation, random_func, eval_func)

        # P+Q
        population.extend(offspring)

        # Environmental Selection
        population = environmental_selection(population, pop_size)
        
        # Plot
        logger.info('Gen:{}'.format(g))
        # plot
        objs1 = [p.obj[0] for p in population]
        objs2 = [p.obj[1] for p in population]
        ax.clear()
        ax.plot(objs1, objs2, 'bo')
        plt.pause(0.0002)
        
        # Save to file
        pkl_path = "{}/population-{}.pkl".format(ec_save_dir, g)
        with open(pkl_path, 'wb') as f:
            pickle.dump(population, f)
        plt.figure()
        plt.plot(objs1, objs2, 'bo')
        plt.savefig('{}/result-{}.png'.format(ec_save_dir, g))
        plt.close('all')
        
main()