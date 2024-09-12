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
        
        self.position = None  # 记录位置和时间戳

class TransferMachine:
    def __init__(self, id):
        self.machine_id = id  # 横移机ID
        self.current_car = None  # 当前正在运送的车身

    def load_car(self, car):
        self.current_car = car

    def unload_car(self):
        car = self.current_car
        self.current_car = None
        return car

    def is_idle(self):
        return self.current_car is None

class PBS:
    def __init__(self, num_cars):
        self.reception_machine = TransferMachine('reception')  # 接车横移机
        self.shipment_machine = TransferMachine('shipment')  # 送车横移机
        self.incar_lane = np.full((6, 10), None)
        self.back_lane = np.full((10), None)

        self.incoming_queue = []
        self.pbs_state = []  # 记录状态
        self.time_passed = 0
        self.num_cars = num_cars

    def add_car_to_queue(self, car):
        self.incoming_queue.append(car)

    def update_position(self, car, position, timestamp):
        car.position = (position, timestamp)
        while len(self.pbs_state) <= timestamp:
            self.pbs_state.append([None] * self.num_cars)
        self.pbs_state[timestamp][car.car_id - 1] = position

    def move_car_to_lane(self, car, lane_id, position, timestamp):
        self.time_passed += [18, 12, 6, 0, 12, 18][lane_id]  # 加上横移机移动时间
        if self.incar_lane[lane_id, position] is None:
            self.incar_lane[lane_id, position] = car
            self.update_position(car, 100 + lane_id * 10 + position, timestamp + self.time_passed)

    def move_car_to_return_lane(self, car, position, timestamp):
        self.time_passed += [24, 18, 12, 6, 12, 18][position]  # 加上横移机移动时间
        if self.back_lane[position] is None:
            self.back_lane[position] = car
            self.update_position(car, 200 + position, timestamp + self.time_passed)

    def handle_reception(self, timestamp):
        if self.reception_machine.is_idle():
            if self.back_lane[9] is not None:
                # 优先处理返回道10停车位上的车身
                car = self.back_lane[9]
                self.back_lane[9] = None
                lane_id = self.assign_lane(car)
                self.move_car_to_lane(car, lane_id, 9, timestamp)
            else:
                for car in self.incoming_queue:
                    lane_id = self.assign_lane(car)
                    if lane_id is not None:
                        self.move_car_to_lane(car, lane_id, 9, timestamp)
                        self.incoming_queue.remove(car)
                        break

    def handle_shipment(self, timestamp):
        if self.shipment_machine.is_idle():
            for lane_id in range(6):
                if self.incar_lane[lane_id, 0] is not None:
                    car = self.incar_lane[lane_id, 0]
                    self.incar_lane[lane_id, 0] = None
                    if self.need_reorder(car):
                        self.move_car_to_return_lane(car, 0, timestamp)
                    else:
                        self.update_position(car, 300, timestamp + [18, 12, 6, 0, 12, 18][lane_id])
                    break

    def assign_lane(self):
        # 随机分配车道
        available_lanes = [i for i in range(6) if self.incar_lane[i, 9] is None]
        if available_lanes:
            return random.choice(available_lanes)
        return available_lanes

    def need_reorder(self, car):
        # 随机决定是否需要调序
        return random.choice([True, False])

    def simulate_dispatching(self, incoming_cars: list[Car], end_time):
        for car in incoming_cars:
            self.incoming_queue.append(car)

        timestamp = 0
        while timestamp <= end_time:
            self.handle_reception(timestamp)
            self.handle_shipment(timestamp)
            self.move_cars_in_lanes(timestamp)
            self.move_cars_in_return_lane(timestamp)
            timestamp += 1

        return self.pbs_state

        return self.pbs_state

    def move_cars_in_lanes(self, timestamp):
        for lane_id in range(6):
            for position in range(8, -1, -1):  # 处理每条车道，从后向前
                if self.incar_lane[lane_id, position] is not None and self.incar_lane[lane_id, position + 1] is None:
                    car = self.incar_lane[lane_id, position]
                    self.incar_lane[lane_id, position] = None
                    self.incar_lane[lane_id, position + 1] = car
                    self.update_position(car, 100 + lane_id * 10 + position + 1, timestamp + 9)  # 车移动需要9秒

    def move_cars_in_return_lane(self, timestamp):
        for position in range(8, -1, -1):  # 从后向前
            if self.back_lane[position] is not None and self.back_lane[position + 1] is None:
                car = self.back_lane[position]
                self.back_lane[position] = None
                self.back_lane[position + 1] = car
                self.update_position(car, 200 + position + 1, timestamp + 9)  # 车移动需要9秒

def animate(state):
    plt.clf()
    plt.title(f"Timestamp: {state}")
    for lane in range(6):
        for position in range(10):
            car = pbs.incar_lane[lane, position]
            if car:
                plt.scatter(lane, position, color='blue', s=100)
    for position in range(10):
        car = pbs.back_lane[position]
        if car:
            plt.scatter(6, position, color='red', s=100)
    plt.xlim(-1, 7)
    plt.ylim(-1, 10)
    plt.pause(0.1)

if __name__ == '__main__':
    # 示例数据
    incars = read_data('附件1.xlsx')

    incoming_cars = [Car(incar["进车顺序"], incar["车型"], incar["动力"], incar["驱动"]) for i, incar in incars.iterrows()]

    # 创建PBS实例
    pbs = PBS(len(incoming_cars))

    end_time = (18*2+90)*318

    # 模拟调度过程
    output_states = pbs.simulate_dispatching(incoming_cars, end_time)

    fig = plt.figure()
    ani = FuncAnimation(fig, animate, frames=len(output_states), repeat=False, )
    plt.show()

    # 输出出车顺序
    for state in output_states:
        if state:
            final_position = state[-1][0]
            if final_position >= 300:
                print(f"Car {incoming_cars[output_states.index(state)].car_id} 出车顺序: {output_states.index(state) + 1}")
