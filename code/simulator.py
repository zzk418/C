import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from q1 import *

class Car:
    def __init__(self, id, model, power, drive):
        self.id = id  # 车身ID
        self.model = model  # 车型
        self.power = power  # 动力类型
        self.drive = drive  # 驱动类型

        # 保存时间戳和位置信息
        self.time = []
        self.position = []

    def update_state(self, time, position):
        # 更新汽车的位置信息和对应的全局时间
        self.time.append(time)
        self.position.append(position)

class TransferMachine:
    def __init__(self, id):
        self.machine_id = id  # 横移机ID
        self.current_car = None  # 当前正在运送的车身
        self.idle = True
        self.position = 'middle'  # 初始位置为中间位置
        self.next_available_time = 0  # 下次可用时间

    def load_car(self, car):
        self.current_car = car
        self.idle = False

    def unload_car(self):
        self.current_car = None
        self.idle = True

class PBS:
    def __init__(self, incars: list):
        self.incars = incars # 进车序列
        self.outcars = [None] * len(incars)# 保存出车序列

        self.in_machine = TransferMachine('in')  # 接车横移机
        self.out_machine = TransferMachine('out')  # 送车横移机
        self.incar_lane = np.full((6, 10), None) # 进车道
        self.back_lane = np.full((10,), None) # 返回道

        self.global_timer = 0  # 全局计时器
        self.hybrid_count = 0
        self.four_by_four_count = 0
        self.two_by_two_count = 0
        self.back_lane_usage = 0
    
    def simulator(self):
        priority_order = np.array([4, 3, 2, 5, 1, 6]) - 1 # 优先顺序车道号

        sim_flag = True
        while sim_flag:
            # 在每一轮模拟中将全局时间加1
            self.global_timer += 1

            # 处理车身移动
            if self.global_timer % 9 == 0:
                self.move_cars()

            # 处理接车横移机任务
            if self.in_machine.idle and self.global_timer >= self.in_machine.next_available_time:
                self.handle_in_machine(priority_order)

            # 处理送车横移机任务
            if self.out_machine.idle and self.global_timer >= self.out_machine.next_available_time:
                self.handle_out_machine(priority_order)

            # 检查所有汽车是否全部送出, 全送出结束循环
            out_count = sum(1 for car in self.outcars if car is not None and car.position[-1] == 'out')
            sim_flag = out_count < len(self.outcars) 

    def move_cars(self):
        # 移动进车道中的车
        for lane in range(6):
            for spot in range(0, 9):
                if self.incar_lane[lane][spot] is None and self.incar_lane[lane][spot + 1] is not None:
                    car = self.incar_lane[lane][spot + 1]
                    self.incar_lane[lane][spot] = car
                    self.incar_lane[lane][spot + 1] = None
                    car.update_state(self.global_timer, (lane, spot))
                    self.outcars[car.id - 1] = car

        # 移动返回道中的车
        for spot in range(1, 10):
            if self.back_lane[spot] is None and self.back_lane[spot - 1] is not None:
                car = self.back_lane[spot - 1]
                self.back_lane[spot] = car
                self.back_lane[spot - 1] = None
                car.update_state(self.global_timer, ('back', spot))
                self.outcars[car.id - 1] = car

    def handle_in_machine(self, priority_order):
        # in区：入车队列进车, 按优先级贪心选择车道10停车位
        if self.incars:
            car = copy.deepcopy(self.incars.pop(0))
            self.in_machine.load_car(car)
            car.update_state(self.global_timer, 'in')
            self.in_machine.unload_car()
            for lane in priority_order:
                if not self.incar_lane[lane][9]:
                    choice_lane = lane
                    break
            time_cost = [18, 12, 6, 0, 12, 18][choice_lane]
            car.update_state(self.global_timer + time_cost, (choice_lane, 9))
            self.in_machine.next_available_time = self.global_timer + time_cost
            self.outcars[car.id - 1] = car
            self.incar_lane[choice_lane][9] = car

        # in区：当返回道10停车位有车身，同时接车横移机空闲时，优先处理返回道10停车位上的车身
        elif self.back_lane[9]:
            car = self.back_lane[9]
            self.in_machine.load_car(car)
            car.update_state(self.global_timer, 'in')
            self.in_machine.unload_car()
            for lane in priority_order:
                if not self.incar_lane[lane][9]:
                    choice_lane = lane
                    break
            time_cost = [24, 18, 12, 6, 12, 18][choice_lane]
            car.update_state(self.global_timer + time_cost, (choice_lane, 9))
            self.in_machine.next_available_time = self.global_timer + time_cost
            self.outcars[car.id - 1] = car
            self.back_lane[9] = None

    def handle_out_machine(self, priority_order):
        # out区：当若干进车道1停车位有车身等候，同时送车横移机空闲时，优先处理最先到达1停车位的车身
        cars = [
            self.incar_lane[i][0]
            for i in range(self.incar_lane.shape[0]) 
            if self.incar_lane[i][0]
        ]

        if cars:
            cars = sorted(cars, key=lambda car: car.time[-1])
            for car in cars:
                self.out_machine.load_car(car)
                self.out_machine.unload_car()
                self.incar_lane[car.position[-1][0]][car.position[-1][1]] = None
                time_cost = [18, 12, 6, 0, 12, 18][car.position[-1][0]]
                car.update_state(self.global_timer + time_cost, 'out')
                self.out_machine.next_available_time = self.global_timer + time_cost
                self.outcars[car.id - 1] = car

        # 约束：如果任意进车道1停车位有车身，那么送车横移机不能设置为空闲状态
        for i in range(self.incar_lane.shape[0]):
            if self.incar_lane[i][0]:
                self.out_machine.idle = False

    def compute_scores(self, car, lane):
        score = 100

        # 计算混动车型间隔
        if car.power == '混动':
            self.hybrid_count += 1
        non_hybrid_count = len([c for c in self.incars if c.power != '混动'])
        if self.hybrid_count != 0 and non_hybrid_count / self.hybrid_count != 2:
            score -= 1 * 0.4

        # 计算四驱和两驱比例
        if car.drive == '四驱':
            self.four_by_four_count += 1
        elif car.drive == '两驱':
            self.two_by_two_count += 1
        if self.four_by_four_count != 0 and self.two_by_two_count / self.four_by_four_count != 1:
            score -= 1 * 0.3

        # 返回道使用次数
        if lane == 'back':
            self.back_lane_usage += 1
            score -= self.back_lane_usage * 0.2

        return score

if __name__ == '__main__':
    incars = read_data('附件1.xlsx')

    incars = [Car(incar["进车顺序"], incar["车型"], incar["动力"], incar["驱动"]) for i, incar in incars.iterrows()]

    # 创建PBS实例
    pbs = PBS(incars)
    pbs.simulator()

    # Print car movement history
    for car in pbs.outcars:
        print(f'Car {car.id} movement history:')
        for t, pos in zip(car.time, car.position):
            print(f'  Time {t}: Position {pos}')
