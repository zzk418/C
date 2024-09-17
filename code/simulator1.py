import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 假设 q1.py 中有定义的 read_data 函数
from q1 import read_data

# 车辆类
class Car:
    def __init__(self, id, model, power, drive):
        self.id = id
        self.model = model
        self.power = power
        self.drive = drive
        self.time = []
        self.position = []

    def update_state(self, time, position):
        self.time.append(time)
        self.position.append(position)

# 横移机类
class TransferMachine:
    def __init__(self, id):
        self.machine_id = id
        self.current_car = None
        self.idle = True
        self.position = 'middle'
        self.next_available_time = 0

    def load_car(self, car):
        self.current_car = car
        self.idle = False

    def unload_car(self):
        self.current_car = None
        self.idle = True

# PBS 仿真类
class PBS:
    def __init__(self, incars: list):
        self.incars = incars
        self.outcars = [None] * len(incars)
        self.in_machine = TransferMachine('in')
        self.out_machine = TransferMachine('out')
        self.incar_lane = np.full((6, 10), None)
        self.back_lane = np.full((10,), None)
        self.global_timer = 0
        self.hybrid_count = 0
        self.four_by_four_count = 0
        self.two_by_two_count = 0
        self.back_lane_usage = 0

    def simulator(self):
        priority_order = np.array([4, 3, 2, 5, 1, 6]) - 1
        sim_flag = True
        while sim_flag:
            self.global_timer += 1
            if self.global_timer % 9 == 0:
                self.move_cars()
            if self.in_machine.idle and self.global_timer >= self.in_machine.next_available_time:
                self.handle_in_machine(priority_order)
            if self.out_machine.idle and self.global_timer >= self.out_machine.next_available_time:
                self.handle_out_machine(priority_order)
            out_count = sum(1 for car in self.outcars if car is not None and car.position[-1] == 'out')
            sim_flag = out_count < len(self.outcars)

    def move_cars(self):
        for lane in range(6):
            for spot in range(0, 9):
                if self.incar_lane[lane][spot] is None and self.incar_lane[lane][spot + 1] is not None:
                    car = self.incar_lane[lane][spot + 1]
                    self.incar_lane[lane][spot] = car
                    self.incar_lane[lane][spot + 1] = None
                    car.update_state(self.global_timer, (lane, spot))
                    self.outcars[car.id - 1] = car
        for spot in range(1, 10):
            if self.back_lane[spot] is None and self.back_lane[spot - 1] is not None:
                car = self.back_lane[spot - 1]
                self.back_lane[spot] = car
                self.back_lane[spot - 1] = None
                car.update_state(self.global_timer, ('back', spot))
                self.outcars[car.id - 1] = car

    def handle_in_machine(self, priority_order):
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
        cars = [self.incar_lane[i][0] for i in range(self.incar_lane.shape[0]) if self.incar_lane[i][0]]
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
        for i in range(self.incar_lane.shape[0]):
            if self.incar_lane[i][0]:
                self.out_machine.idle = False

# 动画仿真
def animate_pbs(pbs):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # 初始化
    max_time = pbs.global_timer

    axcolor = 'lightgoldenrodyellow'
    ax_time = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

    time_slider = Slider(ax_time, 'Time', 0, max_time, valinit=0, valstep=1)

    def update(val):
        frame = int(time_slider.val)
        ax.clear()
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 10)
        ax.set_title(f"Time: {frame}")

        # 绘制进车道
        for car in pbs.outcars:
            if frame in car.time:
                idx = car.time.index(frame)
                pos = car.position[idx]
                if isinstance(pos, tuple):
                    ax.text(pos[0], pos[1], f'{car.id}', ha='center', va='center')
                elif pos == 'out':
                    ax.text(5.5, idx, f'{car.id}', ha='center', va='center', color='red')

    time_slider.on_changed(update)

    # 前进按钮
    axprev = plt.axes([0.8, 0.025, 0.1, 0.04])
    btn_next = Button(axprev, 'Next', color=axcolor, hovercolor='0.975')

    def next(event):
        current_time = time_slider.val
        if current_time < max_time:
            time_slider.set_val(current_time + 1)

    btn_next.on_clicked(next)

    # 后退按钮
    axnext = plt.axes([0.7, 0.025, 0.1, 0.04])
    btn_prev = Button(axnext, 'Prev', color=axcolor, hovercolor='0.975')

    def prev(event):
        current_time = time_slider.val
        if current_time > 0:
            time_slider.set_val(current_time - 1)

    btn_prev.on_clicked(prev)

    plt.show()

if __name__ == '__main__':
    incars = read_data('附件1.xlsx')
    incars = [Car(incar["进车顺序"], incar["车型"], incar["动力"], incar["驱动"]) for i, incar in incars.iterrows()]

    pbs = PBS(incars)
    pbs.simulator()
    animate_pbs(pbs)
