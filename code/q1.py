import pandas as pd
import numpy as np
from typing import List, Dict

# 数据处理
def read_data(file_path: str) -> pd.DataFrame:
    """
    读取车辆信息，返回一个包含车辆数据的列表。

    Args:
        file_path (str): 输入文件路径。

    Returns:
        List[Dict]: 车辆信息列表，每个字典包含车辆的属性。
    """
    # 使用pandas读取CSV文件
    vehicle_df = pd.read_excel(file_path)
    
    return vehicle_df

def write_output_data(result_matrix: np.ndarray, file_path: str) -> None:
    """
    将结果矩阵写入指定的Excel文件。

    Args:
        result_matrix (np.ndarray): 结果矩阵。
        file_path (str): 输出文件路径。
    """
    # 将结果矩阵转换为DataFrame并写入Excel文件
    df = pd.DataFrame(result_matrix)
    df.to_excel(file_path, index=False)


# PSO模块
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = float('inf')

class PSO:
    def __init__(self, num_particles, num_iterations, fitness_function):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.fitness_function = fitness_function
        self.particles = self.initialize_particles()
        self.global_best_position = 0
        self.global_best_score = float('inf')

    def initialize_particles(self) -> List[Particle]:
        """
        初始化粒子群。
        """
        particles = []
        for _ in range(self.num_particles):
            position = np.random.rand() * 10  # 位置初始化为0到10之间的随机数
            velocity = np.random.rand() * 2 - 1  # 速度初始化为-1到1之间的随机数
            particles.append(Particle(position, velocity))
        return particles
    
    def apply_constraints(self, position):
        """
        应用约束条件，将位置限制在[0, 10]范围内。

        Args:
            position (np.ndarray): 粒子的位置。

        Returns:
            np.ndarray: 满足约束条件的位置。
        """
        return np.clip(position, 0, 10)

    def update_particles(self):
        """
        更新粒子的位置和速度。
        """
        w = 0.5  # 惯性权重
        c1 = 1.5  # 个体加速常数
        c2 = 1.5  # 社会加速常数

        for particle in self.particles:
            r1, r2 = np.random.rand(), np.random.rand()

            # 更新速度
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (self.global_best_position - particle.position))

            # 更新位置
            particle.position += particle.velocity

            # 应用约束条件
            particle.position = self.apply_constraints(particle.position)

            # 计算适应度
            current_score = self.fitness_function(particle.position)

            # 更新个体最优
            if current_score < particle.best_score:
                particle.best_score = current_score
                particle.best_position = particle.position

            # 更新全局最优
            if current_score < self.global_best_score:
                self.global_best_score = current_score
                self.global_best_position = particle.position

    def run(self) -> Particle:
        """
        运行PSO算法，返回最佳粒子。
        """
        for _ in range(self.num_iterations):
            self.update_particles()
        best_particle = min(self.particles, key=lambda p: p.best_score)
        return best_particle

# 调度模块
def initialize_pbs_state(num_vehicles: int, num_time_steps: int) -> np.ndarray:
    """
    初始化PBS状态。

    Args:
        num_vehicles (int): 车辆数量。
        num_time_steps (int): 时间步数量。

    Returns:
        np.ndarray: 初始PBS状态矩阵。
    """
    pbs_state = np.full((num_time_steps, num_vehicles), None)
    return pbs_state

def schedule_pbs(particle_position, pbs_state, vehicle_queue) -> np.ndarray:
    """
    根据粒子的位置进行PBS调度，返回结果矩阵。
    """
    num_time_steps, num_vehicles = pbs_state.shape
    result_matrix = pbs_state.copy()

    for i in range(num_vehicles):
        t = int(particle_position[i])
        if t < num_time_steps:
            result_matrix[t, i] = vehicle_queue.iloc[i]['车型']

def fitness_function(pbs_state: np.ndarray, vehicle_queue: pd.DataFrame, num_time_steps) -> float:
    """
    计算当前PBS状态和车辆队列的适应度值。

    Args:
        particle_position (np.ndarray): 粒子的位置。
        pbs_state (np.ndarray): 当前PBS状态矩阵。
        vehicle_queue (pd.DataFrame): 车辆队列信息。
        num_time_steps (int): 动态时间步数。

    Returns:
        float: 适应度值。
    """
    num_vehicles = pbs_state.shape[1]

    # 1. 混动车型间隔评分
    hybrid_intervals = []
    for i in range(num_vehicles - 2):
        if vehicle_queue.loc[i, '动力'] == '混动' and vehicle_queue.loc[i+1, '动力'] != '混动' and vehicle_queue.loc[i+2, '动力'] != '混动':
            hybrid_intervals.append(1)
        else:
            hybrid_intervals.append(0)
    hybrid_score = sum(hybrid_intervals) / (num_vehicles - 2)

    # 2. 四驱与两驱车型比例评分
    four_wd_count = (vehicle_queue['驱动'] == '四驱').sum()
    two_wd_count = (vehicle_queue['驱动'] == '两驱').sum()
    ratio_score = min(four_wd_count, two_wd_count) / max(four_wd_count, two_wd_count)

    # 3. 返回道使用次数评分
    return_lane_usage = 0
    for t in range(num_time_steps):
        for i in range(num_vehicles):
            if pbs_state[t, i] == '返回道':
                return_lane_usage += 1
    return_lane_score = 1 / (1 + return_lane_usage)

    # 4. 总调度时间评分
    completion_times = np.zeros(num_vehicles)
    max_completion_time = 0
    for t in range(num_time_steps):
        for i in range(num_vehicles):
            if pbs_state[t, i] is not None:
                completion_times[i] = t
                if t > max_completion_time:
                    max_completion_time = t
    total_completion_time = np.sum(completion_times)
    total_time_score = 1 / (1 + total_completion_time)

    # 计算总适应度值
    fitness_value = 0.4 * hybrid_score + 0.3 * ratio_score + 0.2 * return_lane_score + 0.1 * total_time_score

    return fitness_value

def update_pbs_state(pbs_state: np.ndarray, max_completion_time: int) -> np.ndarray:
    """
    根据最大完成时间动态更新PBS状态矩阵的时间步数。

    Args:
        pbs_state (np.ndarray): 当前PBS状态矩阵。
        max_completion_time (int): 最大完成时间。

    Returns:
        np.ndarray: 更新后的PBS状态矩阵。
    """
    num_vehicles = pbs_state.shape[1]
    updated_pbs_state = np.full((max_completion_time + 1, num_vehicles), None)
    updated_pbs_state[:pbs_state.shape[0], :] = pbs_state
    return updated_pbs_state

def main():
    # 读取数据
    file_path = '附件1.xlsx'
    vehicle_df = read_data(file_path)

    # 初始化PBS状态
    initial_estimated_time_steps = 1000  # 初步估计时间步数
    pbs_state = initialize_pbs_state(num_vehicles=vehicle_df.shape[0], num_time_steps=initial_estimated_time_steps)

    # 定义适应度函数
    fitness_function(pbs_state, vehicle_df, initial_estimated_time_steps)

    # 运行粒子群算法
    pso = PSO(num_particles=30, num_iterations=1000, fitness_function=fitness_function)
    best_particle = pso.run()

    # 生成调度结果
    result_matrix = schedule_pbs(best_particle.position, pbs_state, vehicle_df)

    # 输出结果
    write_output_data(result_matrix, 'result1.xlsx')


if __name__ == '__main__':
    main()