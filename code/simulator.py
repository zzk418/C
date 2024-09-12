import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from q1 import *

class Car:
    def __init__(self, car_id, model, power, drive):
        self.car_id = car_id  # 车身ID
        self.model = model  # 车型
        self.power = power  # 动力类型
        self.drive = drive  # 驱动类型

        # 保存时间戳位置
        self.time = []
        self.position = []

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

        self.time = 0  # 当前时间戳
        self.end_time = (18*2 + 90) *318
        self.incar_lane_times = [[0] * 10 for _ in range(6)] # 进车道时间记录
        self.back_lane_times = [0] * 10 # 返回道时间记录

    def move_cars(self):
        # 移动进车道中的车
        for lane in range(6):
            for spot in range(9, 0, -1):
                if self.incar_lane[lane][spot] is None and self.incar_lane[lane][spot - 1] is not None:
                    car = self.incar_lane[lane][spot - 1]
                    self.incar_lane[lane][spot] = car
                    self.incar_lane[lane][spot - 1] = None
                    car.position.append((lane, spot))
                    car.time.append(self.time)
                    self.incar_lane_times[lane][spot] = self.time + 9

        # 移动返回道中的车
        for spot in range(9, 0, -1):
            if self.back_lane[spot] is None and self.back_lane[spot - 1] is not None:
                car = self.back_lane[spot - 1]
                self.back_lane[spot] = car
                self.back_lane[spot - 1] = None
                car.position.append(('back', spot))
                car.time.append(self.time)
                self.back_lane_times[spot] = self.time + 9

    def simulator(self):
        while self.time < self.end_time:
            # 处理返回道10停车位的车
            if self.back_lane[9] is not None and self.reception_machine.idle:
                car = self.back_lane[9]
                self.reception_machine.load_car(car)
                self.back_lane[9] = None
                car.time.append(self.time)
                car.position.append('reception_machine')

                # 将车移动到进车道10停车位
                for lane in range(6):
                    if self.incar_lane[lane][9] is None:
                        self.incar_lane[lane][9] = self.reception_machine.unload_car()
                        self.time += [24, 18, 12, 6, 12, 18][lane]
                        self.incar_lane_times[lane][9] = self.time
                        car.time.append(self.time)
                        car.position.append((lane, 9))
                        break

            # 处理进车序列中的车
            if self.incars and self.reception_machine.idle:
                car = self.incars.pop(0)
                self.reception_machine.load_car(car)
                car.time.append(self.time)
                car.position.append('reception_machine')

                # 将车移动到进车道10停车位
                for lane in range(6):
                    if self.incar_lane[lane][9] is None:
                        self.incar_lane[lane][9] = self.reception_machine.unload_car()
                        self.time += [18, 12, 6, 0, 12, 18][lane]
                        self.incar_lane_times[lane][9] = self.time
                        car.time.append(self.time)
                        car.position.append((lane, 9))
                        break

            # 处理进车道1停车位的车
            for lane in range(6):
                if self.incar_lane[lane][0] is not None and self.shipment_machine.idle:
                    car = self.incar_lane[lane][0]
                    self.shipment_machine.load_car(car)
                    self.incar_lane[lane][0] = None
                    car.time.append(self.time)
                    car.position.append('shipment_machine')

                    # 将车移动到总装接车口
                    self.time += [18, 12, 6, 0, 12, 18][lane]
                    self.shipment_machine.unload_car()
                    car.time.append(self.time)
                    car.position.append('output')
                    self.outcars.append(car)
                    break

            self.move_cars()
            self.time += 1
            print(f"Time:{self.time}")

if __name__ == '__main__':
    # 示例数据
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


# def animate(state):
#     plt.clf()
#     plt.title(f"Timestamp: {state}")
#     for lane in range(6):
#         for position in range(10):
#             car = pbs.incar_lane[lane, position]
#             if car:
#                 plt.scatter(lane, position, color='blue', s=100)
#     for position in range(10):
#         car = pbs.back_lane[position]
#         if car:
#             plt.scatter(6, position, color='red', s=100)
#     plt.xlim(-1, 7)
#     plt.ylim(-1, 10)
#     plt.pause(0.1)