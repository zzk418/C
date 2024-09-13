import numpy as np
import pandas as pd
from tqdm import tqdm

from q1 import *

class Car:
    def __init__(self, car_id, model, power, drive):
        self.car_id = car_id  # 车身ID
        self.model = model  # 车型
        self.power = power  # 动力类型
        self.drive = drive  # 驱动类型

        # 保存时间戳位置
        self.timer = 0
        self.time = []
        self.position = []

    def update_time(self, t_increment, position):
        self.timer += t_increment
        self.time.append(self.timer)
        self.position.append(position)

class TransferMachine:
    def __init__(self, id):
        self.machine_id = id  # 横移机ID
        self.current_car = None  # 当前正在运送的车身
        self.idle = True

    def load_car(self, car):
        self.current_car = car
        self.idle = False

    def unload_car(self):
        car = self.current_car
        self.current_car = None
        self.idle = True
        return car

class PBS:
    def __init__(self, incars):
        self.incars = incars # 进车序列
        self.outcars = [] # 保存出车序列

        self.reception_machine = TransferMachine('reception')  # 接车横移机
        self.shipment_machine = TransferMachine('shipment')  # 送车横移机
        self.incar_lane = np.full((6, 10), None) # 进车道
        self.back_lane = np.full((10,), None) # 返回道

        self.hybrid_count = 0
        self.four_by_four_count = 0
        self.two_by_two_count = 0
        self.back_lane_usage = 0

    def move_cars(self):
        # 移动进车道中的车
        for lane in range(6):
            for spot in range(9, 0, -1):
                if self.incar_lane[lane][spot] is None and self.incar_lane[lane][spot - 1] is not None:
                    car = self.incar_lane[lane][spot - 1]
                    self.incar_lane[lane][spot] = car
                    self.incar_lane[lane][spot - 1] = None
                    car.update_time(9, (lane, spot))

        # 移动返回道中的车
        for spot in range(9, 0, -1):
            if self.back_lane[spot] is None and self.back_lane[spot - 1] is not None:
                car = self.back_lane[spot - 1]
                self.back_lane[spot] = car
                self.back_lane[spot - 1] = None
                car.update_time(9, ('back', spot))

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

        # 总调度时间
        total_time = car.timer + [18, 12, 6, 0, 12, 18][lane]
        theoretical_time = 9 * len(self.incars) + 72
        time_penalty = 0.01 * (total_time - theoretical_time)
        score -= time_penalty * 0.1

        return score

    def simulator(self):
        priority_order = [4, 3, 2, 5, 1, 6]  # 优先顺序

        total = len(self.incars)
        for i, it in tqdm(enumerate(self.incars), total=total):
            # 处理返回道10停车位的车
            if self.back_lane[9] is not None and self.reception_machine.idle:
                car = self.back_lane[9]
                self.reception_machine.load_car(car)
                self.back_lane[9] = None
                car.update_time(0, 'reception_machine')

                best_lane = None
                for lane in priority_order:
                    lane_index = lane - 1
                    if self.incar_lane[lane_index][9] is None:
                        best_lane = lane_index
                        break

                if best_lane is not None:
                    self.incar_lane[best_lane][9] = self.reception_machine.unload_car()
                    car.update_time([24, 18, 12, 6, 12, 18][best_lane], (best_lane, 9))

            # 处理进车序列中的车
            if self.incars and self.reception_machine.idle:
                car = self.incars.pop(0)
                self.reception_machine.load_car(car)
                car.update_time(0, 'reception_machine')

                best_lane = None
                for lane in priority_order:
                    lane_index = lane - 1
                    if self.incar_lane[lane_index][9] is None:
                        best_lane = lane_index
                        break

                if best_lane is not None:
                    self.incar_lane[best_lane][9] = self.reception_machine.unload_car()
                    car.update_time([18, 12, 6, 0, 12, 18][best_lane], (best_lane, 9))

            # 处理进车道1停车位的车
            for lane in priority_order:
                lane_index = lane - 1
                if self.incar_lane[lane_index][0] is not None and self.shipment_machine.idle:
                    car = self.incar_lane[lane_index][0]
                    self.shipment_machine.load_car(car)
                    self.incar_lane[lane_index][0] = None
                    car.update_time(0, 'shipment_machine')

                    car.update_time([18, 12, 6, 0, 12, 18][lane_index], 'output')
                    self.shipment_machine.unload_car()
                    self.outcars.append(car)
                    break

            self.move_cars()

            if i == 300:
                print(f"self.outcars：{self.outcars}")

if __name__ == '__main__':
    incars = read_data('附件1.xlsx')

    incars = [Car(incar["进车顺序"], incar["车型"], incar["动力"], incar["驱动"]) for i, incar in incars.iterrows()]

    # 创建PBS实例
    pbs = PBS(incars)
    pbs.simulator()

    # Print car movement history
    for car in pbs.outcars:
        print(f'Car {car.car_id} movement history:')
        for t, pos in zip(car.time, car.position):
            print(f'  Time {t}: Position {pos}')
