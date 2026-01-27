import torch


class Population:
     
    def __init__(self, get_pop, num_instances,pop_size, probelm_size, num_parents, selection_pressure ):
        
        
        
        self.individuals = get_pop

        self.B = num_instances
        self.K = pop_size
        self.N = probelm_size
        self.P = num_parents
        self.selection_pressure = selection_pressure

        self.device = get_pop.device

        self.best_length = torch.full((self.B,), float('inf'), device=self.device, dtype=torch.float32)
        # [B*K, N] → [B, K, N] → [:, 0, :] → [B, N]
        self.best_individual = self.individuals.view(self.B, self.K, self.N)[:, 0, :]
        

        self.batch_offset = torch.arange(self.B, device=self.device) * self.K #batch[i] 시작점

        """
        if is_tsplib == True:
            self.base = False #index 1-N
        else:
            self.base = True #index 0-(N-1)
        """
        #self.base =True

        #shared fitness param
        self.alpha = 2
        self.sigma = self.N * 0.2

        #twopoint Xover threshold
        #self.xover_threshold = int(self.N//10) #int(min(10, self.N * 0.02))
        #print(f"\n xover thershold {self.xover_threshold}")


        #onepoint xover 뒷부분 길이 제어
        # point 범위: (N - threshold) ~ (N - 1)
        # 뒷부분 길이가 1 ~ threshold
        if self.N <= 5000:
            xover_threshold = max( int(self.N * 0.01), 10 ) 
        else:
            xover_threshold = 50
        #print(f" \n xover threshold {xover_threshold}")
        self.min_point = self.N -xover_threshold
        self.max_point = self.N - 4

        #tour 수정시 사용하는 변수
        self.used = torch.zeros(self.B, self.N, device=self.device, dtype=torch.bool)
        self.counts = torch.zeros(self.B, self.N, device=self.device, dtype=torch.long)
        self.city_idx = torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1)



        #print(f"\n pop size {self.individuals.size()}, num_instances {self.B}")
    

    def repair_order_batch_for_twopoint_crossover(self, children):
        """
        Order-based TSP tour repair - GPU 최적화 버전
        
        Args:
            children: (batch_size, n) tensor
            base: True면 0 ~ n-1, False면 1 ~ n
        
        Returns:
            repaired: (batch_size, n) tensor, valid tour
        """
        batch_size, n = children.shape
        device = children.device
        
        # 1-indexed → 0-indexed 변환
        if not self.base:
            children_work = children - 1
        else:
            children_work = children.clone()
        
        # ========== Step 1: 출현 횟수 카운트 ==========
        counts = torch.zeros(batch_size, n, device=device, dtype=torch.long)
        children_clamped = children_work.clamp(0, n - 1)
        counts.scatter_add_(1, children_clamped, torch.ones_like(children_clamped, dtype=torch.long))
        
        # ========== Step 2: Repair 필요 여부 ==========
        needs_repair = ((counts > 1) | (counts == 0)).any(dim=1)
        
        if not needs_repair.any():
            return children
        
        repaired = children_work.clone()
        
        # ========== Step 3: 정렬 ==========
        sorted_vals, sort_indices = torch.sort(children_clamped, dim=1)
        
        # ========== Step 4: 연속 중복 탐지 ==========
        # 이전 값과 같은지 체크
        same_as_prev = torch.zeros(batch_size, n, device=device, dtype=torch.bool)
        same_as_prev[:, 1:] = (sorted_vals[:, 1:] == sorted_vals[:, :-1])
        
        # ========== Step 5: 그룹 내 순서 계산 (벡터화) ==========
        # 핵심 트릭: 그룹이 바뀔 때마다 리셋되는 cumsum
        
        # 새 그룹 시작점 마스크 (same_as_prev가 False인 곳)
        new_group = ~same_as_prev  # [batch_size, n]
        
        # 그룹 ID 부여
        group_id = new_group.long().cumsum(dim=1)  # [batch_size, n]
        
        # 각 위치까지의 그룹 내 순서 계산
        # same_as_prev를 cumsum하되, 그룹이 바뀌면 리셋
        
        # 방법: segment cumsum
        # occurrence_in_sorted[i] = 현재 그룹 내에서 몇 번째인지 (0-indexed)
        
        # same_as_prev가 True면 1, False면 0
        # cumsum 후 그룹 시작점에서 빼기
        
        cumsum_same = same_as_prev.long().cumsum(dim=1)  # [batch_size, n]
        
        # 각 그룹 시작점의 cumsum 값 구하기
        # group_start_cumsum[b, g] = 그룹 g 시작 직전의 cumsum 값
        
        # 더 간단한 방법: 각 위치에서 해당 그룹 시작점의 cumsum 값 빼기
        
        # 그룹 시작점 위치에서의 cumsum 값 (시작점은 0이어야 함)
        # new_group이 True인 위치의 cumsum_same 값을 모든 그룹 멤버에게 전파
        
        # 각 그룹의 시작 cumsum 값
        group_start_cumsum = torch.zeros_like(cumsum_same)
        
        # 그룹 시작점에서의 cumsum_same 값 (실제로는 직전 값)
        # 시작점에서 빼야 할 값 = 시작점 직전의 cumsum
        subtract_val = torch.zeros_like(cumsum_same)
        subtract_val[:, 1:] = cumsum_same[:, :-1]  # 한 칸 shift
        subtract_val[:, 0] = 0
        
        # 그룹 시작점에서만 값 유지, 나머지는 0
        subtract_at_start = subtract_val * new_group.long()
        
        # 각 그룹의 모든 멤버에게 시작점의 subtract 값 전파
        # scatter로 그룹별 값 전파
        
        # group_id를 인덱스로 사용하여 각 그룹의 subtract 값 저장
        max_groups = n  # 최대 그룹 수 = n
        group_subtract = torch.zeros(batch_size, max_groups + 1, device=device, dtype=torch.long)
        group_subtract.scatter_(1, group_id, subtract_at_start)
        
        # 각 위치의 그룹에 해당하는 subtract 값 가져오기
        subtract_for_position = torch.gather(group_subtract, 1, group_id)
        
        # 그룹 내 순서 = cumsum - 그룹 시작점의 cumsum
        occurrence_in_sorted = cumsum_same - subtract_for_position
        
        # ========== Step 6: 원래 위치로 복원 ==========
        occurrence_rank = torch.zeros_like(occurrence_in_sorted)
        occurrence_rank.scatter_(1, sort_indices, occurrence_in_sorted)
        
        # ========== Step 7: 대체 수행 (벡터화) ==========
        # repair가 필요한 batch만 처리
        repair_batch_indices = torch.where(needs_repair)[0]
        
        for b in repair_batch_indices:
            # 대체할 위치들 (두 번째 이상 등장)
            replace_mask = occurrence_rank[b] > 0
            # 누락된 도시들
            missing_mask = counts[b] == 0
            
            replace_positions = torch.where(replace_mask)[0]
            missing_cities = torch.where(missing_mask)[0]
            
            num_replace = min(len(replace_positions), len(missing_cities))
            if num_replace > 0:
                repaired[b, replace_positions[:num_replace]] = missing_cities[:num_replace]
        

        
        return repaired

    def repair_order_batch(self, children, point):
        """
        point 기준: 앞부분 고정, 뒷부분만 repair
        완전 벡터화 - 루프 없음
        """
        #초기화
        self.used.zero_()   
        self.counts.zero_()
     

        work = children.clamp(0, self.N - 1)
        
        # ===== Step 1: 위치 마스크 =====
        idx = torch.arange(self.N, device=self.device)
        front_mask = idx < point.unsqueeze(1)
        back_mask = ~front_mask
        
        # ===== Step 2: 앞부분에서 사용된 도시 마킹 =====
        front_vals = torch.where(front_mask, work, torch.zeros_like(work))
        self.used.scatter_(1, front_vals, front_mask)
        
        # ===== Step 3: 뒷부분 값들 중 중복 여부 =====
        is_duplicate = torch.gather(self.used, 1, work) & back_mask
        
        # ===== Step 4: missing 도시 찾기 =====
        self.counts.scatter_add_(1, work, torch.ones_like(work, dtype=torch.long))
        is_missing = (self.counts == 0)
        
        # ===== Step 5: 벡터화된 대체 =====
        dup_order = is_duplicate.long().cumsum(dim=1) * is_duplicate
        miss_order = is_missing.long().cumsum(dim=1) * is_missing
        
        max_miss = is_missing.sum(dim=1).max().item()
        if max_miss == 0:
            return children
        
        miss_map = torch.zeros(self.B, max_miss + 1, device=self.device, dtype=torch.long)
        
        miss_map.scatter_(1, miss_order, self.city_idx * is_missing.long())
        
        # 중복 위치에 missing 도시 할당
        replacement = torch.gather(miss_map, 1, dup_order.clamp(0, max_miss))
        
        # 최종: 중복 위치만 교체
        repaired = torch.where(is_duplicate, replacement, work)

        return repaired 

    def order_to_locus_batch(self, orders_work):
        """
        Order-based → Locus-based 변환 (batch)
        
        Args:
            orders: (batch_size, n) tensor, valid tour
        
        
        Returns:
            locus: (batch_size, n) tensor (0-indexed)
        """

        
        # 다음 도시
        next_cities = torch.roll(orders_work, shifts=-1, dims=1)
        
        # locus 구성
        locus = torch.zeros_like(orders_work)
        locus.scatter_(dim=1, index=orders_work, src=next_cities)
        
        return locus

    def reverse_locus_batch(self,locus):
        """
        Locus 방향 반전 (batch)
        """
        batch_size, n = locus.shape
        device = locus.device
        
        indices = torch.arange(n, device=device).unsqueeze(0).expand(batch_size, -1)
        
        reversed_locus = torch.zeros_like(locus)
        reversed_locus.scatter_(dim=1, index=locus, src=indices)
        
        return reversed_locus


    def normalize_locus_direction_batch(self,locus):
        """
        Locus 방향 정규화 (batch) 
        """
        reversed_locus = self.reverse_locus_batch(locus)
        

        # 도시 0에서 나가는 edge 비교
        use_forward = locus[:, 0] <= reversed_locus[:, 0]
        use_forward = use_forward.unsqueeze(1).expand_as(locus)
        
        return torch.where(use_forward, locus, reversed_locus)
    

    def pairwise_hamming_distance(self,locus_pop):
        """
        Pairwise Hamming distance
        
        Args:
            locus_pop: [K, N] 또는 [B, K, N] tensor
        
        Returns:
            2D 입력: [K, K]
            3D 입력: [B, K, K]
        """
        if locus_pop.dim() == 2:
            a = locus_pop.unsqueeze(1)
            b = locus_pop.unsqueeze(0)
        else:
            a = locus_pop.unsqueeze(2)
            b = locus_pop.unsqueeze(1)
        
        return (a != b).sum(dim=-1)

    def cross_hamming_distance(self, locus_a, locus_b):
        """
        두 그룹 간 cross hamming distance 계산
        
        Args:
            locus_a: [B, M, N] tensor (예: offspring)
            locus_b: [B, K, N] tensor (예: 기존 population)
        
        Returns:
            cross_dist: [B, M, K] tensor
        """
        a = locus_a.unsqueeze(2)  # [B, M, 1, N]
        b = locus_b.unsqueeze(1)  # [B, 1, K, N]

        return (a != b).sum(dim=-1)  # [B, M, K]

    def compute_pairwise_distance_locus(self,orders_pop):
        """
        Order-based population의 pairwise tour distance 계산
        
        Args:
            orders_pop: (batch_size, n) tensor
                        - base=True: 값은 0 ~ n-1
                        - base=False: 값은 1 ~ n
        
        Returns:
            distances: (batch_size, batch_size) tensor
        """
        # Step 1: Order → Locus 변환 (0-indexed로)
        locus_pop = self.order_to_locus_batch(orders_pop)
        
        # Step 2: 방향 정규화 ; 역방향도 커버
        locus_normalized = self.normalize_locus_direction_batch(locus_pop)
  
        
        # Step 3: Pairwise Hamming distance 여기서만 [B,k,N]이 필요.
        self.locus_normalized = locus_normalized.view(self.B, self.K, self.N)
        distances = self.pairwise_hamming_distance(self.locus_normalized)
        
        return distances


    def compute_shared_fitness(self,tour_lengths, hamming_matrix):
        """
        Shared Fitness 계산 (니칭 기반)
        
        Args:
            tour_lengths: (batch_size* pop_size) - 각 해의 tour length
            hamming_matrix: (batch_size, pop_size, pop_size) - pairwise 해밍 거리
            sigma: float - 니칭 반경
            alpha: float - 감쇠 계수 (default 2.0)
        
        Returns:
            shared_fitness: (batch_size, pop_size)
        """
        # Sharing function 계산
        # sh(d) = 1 - (d/sigma)^alpha if d < sigma else 0
        
        # (batch_size, pop_size, pop_size)
        normalized_dist = hamming_matrix / self.sigma
        
        # sh 값 계산
        sh = 1.0 - torch.pow(normalized_dist, self.alpha)
        
        # sigma 이상이면 0
        sh = torch.where(hamming_matrix < self.sigma, sh, torch.zeros_like(sh))
        
        # Niche count: 각 해에 대해 sh 합산
        # (batch_size, pop_size, pop_size) → (batch_size, pop_size)
        niche_count = sh.sum(dim=2)
        
        # 0으로 나누기 방지
        self.niche_count = torch.clamp(niche_count, min=1e-8) #[B,K]
        
        # Shared fitness = tour_length / niche_count
        # 주의: tour_length는 작을수록 좋음 (minimization)
        niche_count = self.niche_count.view(-1)

        
 

        # tour_lengths 정규화 (0~1 범위)
        tl_min = tour_lengths.min()
        tl_max = tour_lengths.max()
        normalized_tl = (tour_lengths - tl_min) / (tl_max - tl_min + 1e-8)
        
        # 작을수록 좋음 → 역수 대신 (1 - normalized) 사용
        shared_fitness = (1.0 - normalized_tl) / niche_count

        #print(f"\n \t shared_fitness {shared_fitness} {shared_fitness.min(), shared_fitness.max()} ")
        
        return shared_fitness
    
    def compute_sharing_values(self, hamming_matrix):
        """
        Hamming matrix에서 sharing 값 계산
        
        Args:
            hamming_matrix: [B, M, K] 또는 [B, K, K]
        
        Returns:
            sh: 동일 shape의 sharing 값
        """
        normalized_dist = hamming_matrix / self.sigma
        sh = 1.0 - torch.pow(normalized_dist, self.alpha)
        sh = torch.where(hamming_matrix < self.sigma, sh, torch.zeros_like(sh))
        return sh

    def extend_niche_count(self, cross_dist, offspring_dist):
        """
        Offspring 추가 시 niche count 증분 업데이트
        
        Args:
            cross_dist: [B, 1, K] - offspring vs 기존
            offspring_dist: [B, 1, 1] - offspring vs offspring
        
        Returns:
            new_niche_count: [B, K+1]
        """
        # sharing 값 계산

        cross_sh = self.compute_sharing_values(cross_dist)  # [B, 1, K]
        offspring_sh = self.compute_sharing_values(offspring_dist)  # [B, 1, 1]
        
        # 기존 개체 업데이트: 각 기존 개체가 offspring들과 공유하는 값 추가
        # cross_sh [B, 2, K] → transpose → [B, K, 1] → sum → [B, K]
        old_niche_update = cross_sh.transpose(1, 2).sum(dim=2)
   
        
        updated_old_niche = self.niche_count + old_niche_update  # [B, K]

        
        # offspring niche 계산
        # offspring vs 기존: [B, 1, K] → sum → [B, 1]
        # offspring vs offspring: [B, 1, 1] → sum → [B, 1]
        new_niche = cross_sh.sum(dim=2) + offspring_sh.sum(dim=2)  # [B, 1]
        
        # 합치기 [B, K+2]
        new_niche_count = torch.cat([updated_old_niche, new_niche], dim=1)
        
        return new_niche_count
    

    def compute_shared_fitness_from_niche(self, tour_lengths, niche_count):
        """
        Niche count로부터 shared fitness 계산
        
        Args:
            tour_lengths: [B*pop_size] 또는 [B, pop_size]
            niche_count: [B, pop_size]
        
        Returns:
            shared_fitness: [B*pop_size]
        """

        

        niche_flat = niche_count.view(-1)
        niche_flat = torch.clamp(niche_flat, min=1e-8)
        
        
        # 정규화
        tl_min = tour_lengths.min()
        tl_max = tour_lengths.max()
        normalized_tl = (tour_lengths - tl_min) / (tl_max - tl_min + 1e-8)

        
        
        # shared fitness (tour length 작을수록 좋음 → 1-normalized -> fitness값은 tour length가 작을수록 좋도록)
        shared_fitness = (1.0 - normalized_tl) / niche_flat

        #print(f"\n \t shared_fitness {shared_fitness} {shared_fitness.min(), shared_fitness.max()} ")
        
        
        return shared_fitness

    def roulette_wheel_selection_wo_shared_fitness(self):
        """
        Args:
            shared_fitness: (batch_size * pop_size,) - 클수록 좋음
            batch_size: int
            pop_size: int
            num_samples: int - 각 배치에서 선택할 부모 수
        
        Returns:
            parent_idx: (batch_size * num_samples,)
        """

        # tour_lengths 정규화 (0~1 범위)
        tl_min = self.tour_length.min()
        tl_max = self.tour_length.max()
        normalized_tl = (self.tour_length - tl_min) / (tl_max - tl_min + 1e-8)

        # 작을수록 좋음 → 역수 대신 (1 - normalized) 사용
        normalized_tl = (1.0 - normalized_tl)

        self.fitness = normalized_tl
        fitness = normalized_tl.view(self.B, self.K)
        
        #선택압 조절
        f_min = fitness.min(dim=1, keepdim=True).values
        f_max = fitness.max(dim=1, keepdim=True).values
        diff = f_max - f_min

        # 모두 동일하면 균등 확률
        is_same = diff < 1e-8
        diff = torch.where(is_same, torch.ones_like(diff), diff)

        normalized = (fitness - f_min) / diff
        scaled_fitness = 1 + (self.selection_pressure - 1) * normalized

        # 동일 fitness면 균등
        scaled_fitness = torch.where(is_same, torch.ones_like(scaled_fitness), scaled_fitness)

        selection_prob = scaled_fitness / scaled_fitness.sum(dim=1, keepdim=True)

        #softmax selection
        #selection_prob = torch.softmax(fitness, dim=1)
        
        
        #print(f"\n selection check")
        #print(f"\n selection_prob {selection_prob.size()}  {selection_prob}")
        

        # (batch_size, num_samples)
        parent_local = torch.multinomial(selection_prob, num_samples=self.P, replacement=False)
        
        # 전체 인덱스로 변환
        parent1_idx = self.batch_offset + parent_local[:,0] #batch_offset은 batch[i] 시작점
        parent2_idx = self.batch_offset + parent_local[:,1]
        
        #print(f"\n\n parent inx {parent1_idx} {parent2_idx}")

        self.parent1 = self.individuals[parent1_idx] #(batch_size , N)
        self.parent2 = self.individuals[parent2_idx]

    def roulette_wheel_selection(self):
        """
        Args:
            shared_fitness: (batch_size * pop_size,) - 클수록 좋음
            batch_size: int
            pop_size: int
            num_samples: int - 각 배치에서 선택할 부모 수
        
        Returns:
            parent_idx: (batch_size * num_samples,)
        """
        fitness = self.fitness.view(self.B, self.K)
        
        #선택얍 조절
        f_min = fitness.min(dim=1, keepdim=True).values
        f_max = fitness.max(dim=1, keepdim=True).values
        diff = f_max - f_min

        # 모두 동일하면 균등 확률
        is_same = diff < 1e-8
        diff = torch.where(is_same, torch.ones_like(diff), diff)

        normalized = (fitness - f_min) / diff
        scaled_fitness = 1 + (self.selection_pressure - 1) * normalized

        # 동일 fitness면 균등
        scaled_fitness = torch.where(is_same, torch.ones_like(scaled_fitness), scaled_fitness)

        selection_prob = scaled_fitness / scaled_fitness.sum(dim=1, keepdim=True)

        #softmax selection
        #selection_prob = torch.softmax(fitness, dim=1)
        
        
        #print(f"\n selection check")
        #print(f"\n selection_prob {selection_prob.size()}  {selection_prob}")
        

        # (batch_size, num_samples)
        parent_local = torch.multinomial(selection_prob, num_samples=self.P, replacement=False)
        
        # 전체 인덱스로 변환
        parent1_idx = self.batch_offset + parent_local[:,0] #batch_offset은 batch[i] 시작점
        parent2_idx = self.batch_offset + parent_local[:,1]
        
        #print(f"\n\n parent inx {parent1_idx} {parent2_idx}")

        self.parent1 = self.individuals[parent1_idx] #(batch_size , N)
        self.parent2 = self.individuals[parent2_idx]
        
    def onepoint_crossover(self):
        # 단일 교차점 생성 (1 ~ N-1 범위)
        # 단일 값 생성 후 배치 크기로 확장        
        
        # point 범위: (N - threshold) ~ (N - 3)

        # point 최대값 = N - 4 → leftover 최소 4
        
        point_val = torch.randint(self.min_point, self.max_point + 1, (1,), device=self.device)
     
        leftover =self.N - point_val #prc가 필요한 길이
        #print(f"\n 남은 길이 : {self.N - point_val}")
        
        point = point_val.expand(self.B)

        
        
        # 마스크: point 이전 = False, point 이후 = True
        idx = torch.arange(self.N, device=self.device)
        mask = idx >= point.unsqueeze(1)  # [B, N]
        
        # 50% 확률로 parent 순서 결정
        swap = (torch.rand(self.B, device=self.device) < 0.5).unsqueeze(1)
        
        offspring = torch.where(
            swap,
            torch.where(mask, self.parent1, self.parent2),
            torch.where(mask, self.parent2, self.parent1)
        )
        
        #print(f"\n parents p1 {self.parent1}")
        #print(f"\n parents p2 {self.parent2}")
        #print(f"\n offspring point : {point}  off sol :  {offspring}")
        offspring = self.repair_order_batch(offspring, point)
        #print(f"\n repaired offspring check in xover {offspring}")
        
        return offspring, leftover

    def twopoint_crossover(self):
        # 2개 점 생성 후 정렬
        points = torch.randint(1, self.N, (self.B, 2), device=self.device)

        #"""
        ##THRESHOLD start
        
        point1 = torch.randint(1, self.N - 1, (self.B,), device=self.device)
        diff = torch.randint(1, self.xover_threshold + 1, (self.B,), device=self.device)
        point2 = torch.clamp(point1 + diff, max=self.N-1)
        ##THRESHOLD end
        #"""


        point1, point2 = points.min(dim=1).values, points.max(dim=1).values
        
        # 단일 마스크 생성 (중간 텐서 없이)
        idx = torch.arange(self.N, device=self.device)
        mask = (idx >= point1.unsqueeze(1)) & (idx < point2.unsqueeze(1))


        # 50% 확률로 parent 순서 결정 (배치별 독립)
        swap = torch.rand(self.B, device=self.device) < 0.5  # [B]
        swap = swap.unsqueeze(1)  # [B, 1] → broadcast용
        
        offspring = torch.where(
            swap,
            torch.where(mask, self.parent1, self.parent2),  # swap=True: offspring2 방식
            torch.where(mask, self.parent2, self.parent1)   # swap=False: offspring1 방식
        )

        offspring = self.repair_order_batch(offspring) # (batch_size , N)
        

        #print(f"\n\n cross over check")
        #print(f"\n parent1 {self.parent1}")
        #print(f"\n parent2 {self.parent2}")
        #print(f"\n xover point {point1, point2}")
        #print(f"\n offspring {offspring}")
  


        return offspring


    def eval_population(self, individuals, tour_length):
        
        self.tour_length = tour_length #[B*K]
        
        #population 내의 individuals 끼리 hamming distance구하기
        self.hamming_dist =self.compute_pairwise_distance_locus(individuals) #[B, K, K]

        #shared fitness 
        self.fitness=self.compute_shared_fitness(self.tour_length, self.hamming_dist) #[B*K]

        #print(f"\n dfsdfsdf shared {self.shared_fit.size()}")
        
        
        # 배치별 최소 인덱스
        local_idx = self.tour_length.view(self.B, self.K).argmin(dim=1)  # (B,)
        
        # 전체 인덱스로 변환
        global_idx = self.batch_offset + local_idx  # (B,) #Batch_offset은 batch[i] 시작점
        
        self.candidate_best_individual = self.individuals[global_idx]  # (B, N)
        self.candidate_best_length = self.tour_length.view(self.B, self.K)[torch.arange(self.B, device=self.device), local_idx]  # (B,)
 
        
        #best length update
        self.get_best()
    

    
    def select_top_k(self):
        K2 = self.K + 1
        
        # 1. reshape
        shared_fit = self.fitness.view(self.B, K2)
        individuals = self.individuals.view(self.B, K2, self.N)
        tour_length = self.tour_length.view(self.B, K2)
        
        # 2. top K 선택
        _, top_indices = torch.topk(shared_fit, k=self.K, dim=1)  # [B, K]

        # 3. 공통 idx 생성 (재사용)
        idx_2d = top_indices.unsqueeze(-1)  # [B, K, 1]
        idx_3d = idx_2d.expand(-1, -1, self.N)   # [B, K, N]

        
        #자식이 포함되는지 확인
        #offspring_included = (top_indices >= self.K).any()
        #if offspring_included:
        #    print(f"\n \t\t offspring newly included! \n \t") # population size {self.K} top index : {top_indices}" )
        


        
        # 4. gather (idx 재사용)
        self.individuals = torch.gather(individuals, dim=1, index=idx_3d).view(self.B * self.K, self.N)
        self.tour_length = torch.gather(tour_length, dim=1, index=top_indices).view(self.B * self.K)
        self.fitness = torch.gather(shared_fit, dim=1, index=top_indices).view(self.B * self.K)
        self.niche_count = torch.gather(self.niche_count.view(self.B, K2), dim=1, index=top_indices)
        self.locus_normalized = torch.gather(self.locus_normalized.view(self.B, K2, self.N), dim=1, index=idx_3d)
        
        # 5. hamming: 행/열 선택
        idx_row = idx_2d.expand(-1, -1, K2)  # [B, K, K+2]
        selected_hamming = torch.gather(self.hamming_dist, dim=1, index=idx_row)
        
        idx_col = top_indices.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, K]
        self.hamming_dist = torch.gather(selected_hamming, dim=2, index=idx_col)
    
    def select_top_k_wo_shared(self): #_wo_shared_fit
        K2 = self.K + 1
        
        # 1. reshape
        shared_fit = self.tour_length.view(self.B, K2)
        individuals = self.individuals.view(self.B, K2, self.N)
        tour_length = self.tour_length.view(self.B, K2)
        
        # 2. top K 선택
        _, top_indices = torch.topk(shared_fit, k=self.K, dim=1)  # [B, K]

        """
        #자식이 포함되는지 확인
        offspring_included = (top_indices >= self.K).any()
        if offspring_included:
            #print(f"\n \t\t offspring newly included! \n \t") # population size {self.K} top index : {top_indices}" )
        """ 
        # 3. 공통 idx 생성 (재사용)
        idx_2d = top_indices.unsqueeze(-1)  # [B, K, 1]
        idx_3d = idx_2d.expand(-1, -1, self.N)   # [B, K, N]
        
        # 4. gather (idx 재사용)
        self.individuals = torch.gather(individuals, dim=1, index=idx_3d).view(self.B * self.K, self.N)
        self.tour_length = torch.gather(tour_length, dim=1, index=top_indices).view(self.B * self.K)
        self.fitness = torch.gather(shared_fit, dim=1, index=top_indices).view(self.B * self.K)
       
      


    def select_top_k_backup(self):
        """
        Shared fitness 기준 top K개 선택
        
        Args:
            shared_fit: [B*(K+2)] - fitness 값
            individuals: [B*(K+2), N] - 개체들
            k: 선택할 개수 (default: self.K)
        
        Returns:
            selected_individuals: [B*K, N]
            selected_fitness: [B*K]
            top_indices: [B, K] - 선택된 인덱스
        """
       

        # 1. reshape to [B, pop_size+2, ...]
        shared_fit = self.fitness.view(self.B, self.K+2)
        individuals = self.individuals.view(self.B, self.K+2, self.N)
        tour_length = self.tour_length.view(self.B, self.K+2)
        
        # 2. 각 B마다 top K개 선택
        _, top_indices = torch.topk(shared_fit, k=self.K, dim=1)  # [B, K]


        
        # 3. gather로 선택
        # individuals: [B, K+2, N] → [B, K, N]
        idx = top_indices.unsqueeze(-1).expand(-1, -1, self.N)  # [B, K, N]
        selected_individuals = torch.gather(individuals, dim=1, index=idx)
        selected_tour_length = torch.gather(tour_length, dim=1, index=top_indices)  # [B, K]
        self.niche_count = torch.gather(self.niche_count, dim=1, index=top_indices)  # [B, K]  
        self.locus_normalized = torch.gather(self.locus_normalized , dim=1, index=idx)  # [B, K,N]



        # hamming_dist: [B, K+2, K+2] → [B, K, K]
        # 행 선택: [B, K+2, K+2] → [B, K, K+2]
        idx_row = top_indices.unsqueeze(-1).expand(-1, -1, self.K+2)  # [B, K, K+2]

   
        selected_hamming = torch.gather(self.hamming_dist, dim=1, index=idx_row)  # [B, K, K+2]


        # 열 선택: [B, K, K+2] → [B, K, K]
        idx_col = top_indices.unsqueeze(1).expand(-1,self.K, -1)  # [B, K, K]

        
        self.hamming_dist= torch.gather(selected_hamming, dim=2, index=idx_col)  # [B, K, K]
 
 
        # fitness도 선택
        selected_fitness = torch.gather(shared_fit, dim=1, index=top_indices)  # [B, K]


        
        # 4. flatten
        self.individuals = selected_individuals.view(self.B * self.K, self.N)
        self.fitness = selected_fitness.view(self.B * self.K)
        self.tour_length = selected_tour_length.view(self.B * self.K)


    

    def replace(self, offspring_sol, offspring_length): #including evaluation
        #concat_offspring
        
        # (B, K, N)으로 reshape
        ind_reshaped = self.individuals.view(self.B, self.K, self.N)
        off_reshaped = offspring_sol.view(self.B, 1, self.N)
        
        # 배치별로 concat: (B, K+1, N)
        new_ind = torch.cat([ind_reshaped, off_reshaped], dim=1)
        
        # flatten: (B * (K+1), N)
        self.individuals = new_ind.view(-1, self.N)
        
        # tour_lengths도 동일하게
        tl_reshaped = self.tour_length.view(self.B, self.K)
        ol_reshaped = offspring_length.view(self.B, 1)

        #print(f"\n\n offspring length {ol_reshaped}")
        
        new_tl = torch.cat([tl_reshaped, ol_reshaped], dim=1)  # (B, K+1)

        
        self.tour_length = new_tl.view(-1)  # (B * (K+1),)
        
        #save_hall of fame
        
        # 배치별 최소 인덱스
        local_idx = new_tl.argmin(dim=1)  # (B,)
        
        new_batch_offset = torch.arange(self.B, device=self.device) * (self.K+1)

        
        # 전체 인덱스로 변환
        global_idx = new_batch_offset + local_idx  # (B,) #Batch_offset은 batch[i] 시작점
        
        self.candidate_best_individual = self.individuals[global_idx]  # (B, N)
        self.candidate_best_length = new_tl[torch.arange(self.B, device=self.device), local_idx]  # (B,)
 
        
        #best length update
        self.get_best()
    
    
        
        #update fitness_score
        
        # offspring locus 변환 및 정규화
        offspring_locus = self.order_to_locus_batch(offspring_sol)
        offspring_locus = self.normalize_locus_direction_batch(offspring_locus)
        offspring_locus = offspring_locus.view(self.B, 1, self.N)  # [B, 1, N]
        
        # 1. cross distance: offspring vs 기존 [B, 2, K]
        cross_dist = self.cross_hamming_distance(offspring_locus, self.locus_normalized)


        # 2. offspring 간 distance [B, 1, 1]
        offspring_dist = self.pairwise_hamming_distance(offspring_locus)
        
        # 3. 조합
        # 상단: [B, K, K] + [B, K, 1] → [B, K, K+1]
        top = torch.cat([self.hamming_dist, cross_dist.transpose(1, 2)], dim=2)
        
        # 하단: [B, 1, K] + [B, 1, 1] → [B, 1, K+1]
        bottom = torch.cat([cross_dist, offspring_dist], dim=2)
        
        # 전체: [B, K+1, K+1]
        self.hamming_dist = torch.cat([top, bottom], dim=1)
        
        # locus_normalized도 확장 [B, K+1, N]
        self.locus_normalized = torch.cat([self.locus_normalized, offspring_locus], dim=1)

        #shared fitness 
        self.niche_count = self.extend_niche_count(cross_dist, offspring_dist)
        self.fitness= self.compute_shared_fitness_from_niche(self.tour_length, self.niche_count)
     



        #N개만 population update, 기준 : shared_fitness score or / length...
        self.select_top_k() 

    def replace_wo_shared(self, offspring_sol, offspring_length): #_wo_shared
        #concat_offspring
        
        # (B, K, N)으로 reshape
        ind_reshaped = self.individuals.view(self.B, self.K, self.N)
        off_reshaped = offspring_sol.view(self.B, 1, self.N)
        
        # 배치별로 concat: (B, K+1, N)
        new_ind = torch.cat([ind_reshaped, off_reshaped], dim=1)
        
        # flatten: (B * (K+1), N)
        self.individuals = new_ind.view(-1, self.N)
        
        # tour_lengths도 동일하게
        tl_reshaped = self.tour_length.view(self.B, self.K)
        ol_reshaped = offspring_length.view(self.B, 1)

        #print(f"\n\n offspring length {ol_reshaped}")
        
        new_tl = torch.cat([tl_reshaped, ol_reshaped], dim=1)  # (B, K+1)

        
        self.tour_length = new_tl.view(-1)  # (B * (K+1),)
        
        #save_hall of fame
        
        # 배치별 최소 인덱스
        local_idx = new_tl.argmin(dim=1)  # (B,)
        
        new_batch_offset = torch.arange(self.B, device=self.device) * (self.K+1)

        
        # 전체 인덱스로 변환
        global_idx = new_batch_offset + local_idx  # (B,) #Batch_offset은 batch[i] 시작점
        
        self.candidate_best_individual = self.individuals[global_idx]  # (B, N)
        self.candidate_best_length = new_tl[torch.arange(self.B, device=self.device), local_idx]  # (B,)
 
        
        #best length update
        self.get_best()
    


        #N개만 population update, 기준 : shared_fitness score or / length...
        self.select_top_k() #현재 fitness score  ####여기서 문제발생



    
    def mutation(self):
        BK = self.B * self.K
        
        # 1. inverse + roll 마스크 생성
        inverse_mask = torch.rand(BK, device=self.device) < 0.5  # [B*K]
        roll_shifts = torch.randint(0, self.N, (BK,), device=self.device)
        
        # 2. inverse 적용
        idx = torch.arange(self.N, device=self.device)
        base_idx = torch.where(
            inverse_mask.unsqueeze(1),
            self.N - 1 - idx,
            idx.unsqueeze(0).expand(BK, -1)
        )  # [B*K, N]
        
        # 3. roll 적용
        final_idx = (base_idx - roll_shifts.unsqueeze(1)) % self.N  # [B*K, N]
        
  
        # 4. gather
        self.individuals = torch.gather(self.individuals, dim=1, index=final_idx)
    

    def get_best(self):
        # 각 배치별로 비교하여 더 좋은 값 업데이트
        update_mask = self.candidate_best_length < self.best_length  # [B] boolean 텐서
        
        self.best_length = torch.where(update_mask, self.candidate_best_length, self.best_length)
        
        # individual도 업데이트 (마스크 확장 필요)
        update_mask_expanded = update_mask.unsqueeze(1).expand_as(self.best_individual)  # [B, N]
        self.best_individual = torch.where(update_mask_expanded, self.candidate_best_individual, self.best_individual)

        #print(f"\n get beest best length {self.best_length}")
        return self.best_length
    
    def get_information(self):
        return self.tour_length

