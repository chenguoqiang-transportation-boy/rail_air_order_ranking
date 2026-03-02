import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from typing import Dict, List, Optional, Set
from data_generation import (
    DataLoader, NetworkInitializer, OrderGenerator, ScenarioGenerator,
    DelayModel, Costs, Order, TransportMode, OrderStatus
)
from data_save import ResultsSaver
from cost_calculator import CostCalculator
from optimization_pressure_priority import TwoStageOptimizer
from simulation_logger import SimulationLogger  # 假设这个文件存在

class SimulationState:
    """存储仿真的所有状态数据，管理订单在不同阶段的池子"""
    def __init__(self):
        self.current_time = 0.0
        self.orders: Dict[str, Order] = {}
        self.trains = {}
        self.flights = {}

        # 订单集合 (使用 ID)
        self.pool_new: List[str] = []  # 新到达订单
        self.pool_waiting_for_train: set = set()  # 等待上火车
        self.pool_on_train: set = set()  # 在火车上
        self.pool_transferring: set = set()  # 换装中
        self.pool_waiting_for_flight: set = set()  # 等待登机
        self.pool_on_flight: set = set()  # 在飞机上
        self.pool_finished: List[str] = []  # 已完成

        # 构建索引（根据订单起始点找寻可以使用的列车和航班）
        self.train_origin_idx: Dict[str, Set[str]] = defaultdict(set)
        self.flight_dest_idx: Dict[str, Set[str]] = defaultdict(set)

    def get_order(self, oid: str) -> Order:
        return self.orders[oid]

    def add_new_orders(self, orders: List[Order]):
        for o in orders:
            self.orders[o.id] = o
            self.pool_new.append(o.id)


class TransportationEngine:
    """负责物理世界的状态演进，按时间顺序处理"""
    def __init__(self, state: SimulationState, cost_calc: CostCalculator):
        self.state = state
        self.cost_calc = cost_calc

    def process_events(self):
        t = self.state.current_time

        # ==========================================
        # 1. 上车逻辑(WAITING -> ON_TRAIN)：等待火车池 → 火车池（到达上车时间）
        # ==========================================
        for oid in list(self.state.pool_waiting_for_train):
            order = self.state.get_order(oid)
            if order.assigned_train and t >= order.actual_pickup_time:
                order.status = OrderStatus.ON_TRAIN
                self.state.pool_waiting_for_train.remove(oid)
                self.state.pool_on_train.add(oid)

        # ==========================================
        # 2. 列车到达 (ON_TRAIN -> TRANSFERRING)：火车池 → 换装池
        # ==========================================
        for oid in list(self.state.pool_on_train):
            order = self.state.get_order(oid)
            train = self.state.trains[order.assigned_train]
            if t >= train.actual_arrival:
                order.status = OrderStatus.TRANSFERRING
                order.actual_transfer_start = train.actual_arrival
                # 【关键赋值】：此时才知道换装完成时间 transfer_complete_time
                order.transfer_duration = DelayModel.sample_transfer_time()
                order.transfer_complete_time = order.actual_transfer_start + order.transfer_duration

                self.state.pool_on_train.remove(oid)
                self.state.pool_transferring.add(oid)

        # ==========================================
        # 3. 换装完成 (TRANSFERRING -> WAITING_FOR_FLIGHT)：换装池 → 等待登机池
        # ==========================================
        for oid in list(self.state.pool_transferring):
            order = self.state.get_order(oid)
            if t >= order.transfer_complete_time:  # 换装完成
                self.state.pool_transferring.remove(oid)
                # 当订单进入pool_transferring后，如果没有分配航班，则视为外包
                if order.assigned_flight is None:
                    order.actual_delivery_time = order.deadline  # 【我还是希望在时间超过order.actual_delivery_time之后再修改状态】
                    order.status = OrderStatus.DELIVERED
                    self.state.pool_finished.append(oid)
                else:
                    # 正常联运：进入等待航班池
                    order.status = OrderStatus.WAITING_FOR_FLIGHT
                    self.state.pool_waiting_for_flight.add(oid)

        # ==========================================
        # 4. 登机 (WAITING_FOR_FLIGHT -> ON_FLIGHT)：等待登机池 → 飞机池
        # ==========================================
        for oid in list(self.state.pool_waiting_for_flight):
            order = self.state.get_order(oid)
            flight = self.state.flights.get(order.assigned_flight)
            if t >= flight.actual_departure:
                order.status = OrderStatus.ON_FLIGHT
                self.state.pool_waiting_for_flight.remove(oid)
                self.state.pool_on_flight.add(oid)

        # ==========================================
        # 5. 航班到达 (ON_FLIGHT -> DELIVERED)：飞机池 → 完成池
        # ==========================================
        for oid in list(self.state.pool_on_flight):
            order = self.state.get_order(oid)
            flight = self.state.flights.get(order.assigned_flight)
            if t >= flight.actual_arrival:
                order.actual_delivery_time = flight.actual_arrival
                order.status = OrderStatus.DELAYED if t > order.deadline else OrderStatus.DELIVERED
                self.state.pool_on_flight.remove(oid)
                self.state.pool_finished.append(oid)

class Simulator:
    def __init__(self, config: Dict, data_dir: str, file_params: Dict[str, str]):
        self.config = config
        self.costs = Costs(**config.get('costs', {}))

        # 配置参数作为实例属性
        self.period_len = config.get('period_length', 30)  # 时间步
        self.horizon = config.get('planning_horizon', 900)  # 规划窗口

        # 核心组件
        self.loader = DataLoader(data_dir)
        self.data = self.loader.load_all(
            train_file=file_params['train'],
            station_file=file_params['train_station'],
            flight_file=file_params['flight'],
            cargo_file=file_params['cargo']
        )

        self.state = SimulationState()
        self._init_network()

        self.cost_calc = CostCalculator(self.costs, self.data['origin'], self.state)
        self.optimizer = TwoStageOptimizer(self.cost_calc)
        self.engine = TransportationEngine(self.state, self.cost_calc)
        self.scenario_gen = ScenarioGenerator(DelayModel())

        # 预加载所有订单，此时订单状态为OrderStatus.NEW
        self.all_orders = OrderGenerator(self.data['cargo']).generate()
        self.all_orders.sort(key=lambda x: x.earliest_pickup)  # 按时间排序以便分发
        # 计算最大仿真时间
        self.max_time = self.all_orders[-1].deadline + 2000
        self.history = []

    def _init_network(self):
        """初始化并注入延误（上帝视角）"""
        self.state.trains = NetworkInitializer.init_trains(
            self.data['train'], self.data['train_station'], self.data['origin'])
        self.state.flights = NetworkInitializer.init_flights(
            self.data['flight'], self.data['destination'])
        self._build_static_indexes()

        # 预计算实际时刻 (注入上帝视角的真实延误)
        dm = DelayModel()
        for t in self.state.trains.values():
            delay_acc = 0
            travel_time_acc = 0
            for i, s in enumerate(t.stations):
                d = dm.sample_train_delay()
                s.station_delay = d
                delay_acc += d
                travel_time_acc += s.travel_time
                s.travel_time_acc = travel_time_acc
                s.actual_arrival = t.sch_departure + travel_time_acc + delay_acc
                # 如果是最后一个站点
                if i == len(t.stations) - 1:
                    t.actual_arrival = s.actual_arrival

        for f in self.state.flights.values():
            d = dm.sample_flight_delay(f.average_delay, f.worst_delay)
            f.actual_departure = f.sch_departure + d
            f.actual_arrival = f.actual_departure + f.travel_time

    def _build_static_indexes(self):
        """一次性构建静态查找索引"""
        # 1. 构建列车索引 (Origin -> Train IDs)
        for t_id, train in self.state.trains.items():
            for s in train.stations:
                self.state.train_origin_idx[s.origin_id].add(t_id)
        # 2. 构建航班索引 (Destination -> Flight IDs)
        for f_id, flight in self.state.flights.items():
            self.state.flight_dest_idx[flight.destination_id].add(f_id)

    def _calculate_utilization(self):
        """计算当前时刻的资源平均利用率"""
        # 火车利用率
        train_utils = []
        for t in self.state.trains.values():
            if t.capacity > 0:
                util = (t.capacity - t.remaining_capacity) / t.capacity
                train_utils.append(util)
        avg_train = sum(train_utils) / len(train_utils) if train_utils else 0

        # 飞机利用率
        flight_utils = []
        for f in self.state.flights.values():
            if f.capacity > 0:
                util = (f.capacity - f.remaining_capacity) / f.capacity
                flight_utils.append(util)
        avg_flight = sum(flight_utils) / len(flight_utils) if flight_utils else 0

        return avg_train, avg_flight

    def _capture_snapshot(self, t, new_orders_count, adjustable_count):
        """捕获当前时间步的统计数据"""

        # 统计当前已完成订单的状态
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]

        num_delivered = 0
        num_delayed = 0
        num_adhoc = 0
        total_cost = 0.0

        for o in finished_orders:
            total_cost += (o.total_cost if o.total_cost else 0)
            if o.mode == TransportMode.AD_HOC or o.status == OrderStatus.AD_HOC_MODE:
                num_adhoc += 1
            elif o.status == OrderStatus.DELAYED:
                num_delayed += 1
            elif o.status == OrderStatus.DELIVERED:
                num_delivered += 1

        total_finished = len(finished_orders)
        delay_rate = num_delayed / total_finished if total_finished > 0 else 0
        adhoc_rate = num_adhoc / total_finished if total_finished > 0 else 0

        snapshot = {
            'time': t,
            'new_orders': new_orders_count,
            'adjustable_orders': adjustable_count,
            # 冻结状态：在火车上 + 换乘中 + 在飞机上
            'frozen_train': len(self.state.pool_on_train) + len(self.state.pool_transferring),
            'frozen_flight': len(self.state.pool_on_flight),
            'metrics': {
                'total_cost': total_cost,
                'num_delivered': num_delivered,
                'num_delayed': num_delayed,
                'num_adhoc': num_adhoc,
                'delay_rate': delay_rate,
                'adhoc_rate': adhoc_rate
            }
        }
        self.history.append(snapshot)

    def run(self):
        print(f"开始仿真: {len(self.all_orders)} 订单, 周期 {self.period_len}min")
        self.history = []

        while self.state.current_time < self.max_time:
            t = self.state.current_time
            print("=================当前时刻:", int(t), "s=================")

            # 1. 物理状态演进
            self.engine.process_events()

            # 2. 新订单到达
            batch = self._get_new_orders(t, self.period_len)
            self.state.add_new_orders(batch)

            # 3. 准备优化输入
            # 筛选出需要决策的订单：新订单 + 等待中的订单
            adjustable_orders = [self.state.get_order(oid) for oid in self.state.pool_new] + \
                            [self.state.get_order(oid) for oid in self.state.pool_waiting_for_train] + \
                            [self.state.get_order(oid) for oid in self.state.pool_waiting_for_flight]

            if adjustable_orders:
                # 筛选可用资源
                avail_trains = {k: v for k, v in self.state.trains.items() if t <= v.sch_departure <= t + self.horizon}
                avail_flights = {k: v for k, v in self.state.flights.items() if t <= v.sch_departure <= t + self.horizon}

                # 准备 Train Actuals (用于场景生成)
                train_actuals = {
                    tid: ([s.station_delay for s in t.stations],
                          [s.actual_arrival for s in t.stations])
                    for tid, t in avail_trains.items()
                }

                opt_input = {
                    'adjustable_orders': adjustable_orders,
                    'available_trains': avail_trains,
                    'available_flights': avail_flights,
                    'indexes': {
                        'train_origin': self.state.train_origin_idx,
                        'flight_dest': self.state.flight_dest_idx
                    },
                    'scenarios': self.scenario_gen.generate(avail_trains, avail_flights,
                                                          self.config.get('num_scenarios', 50), t, train_actuals)
                }

                # 4. 执行优化
                plan = self.optimizer.solve(opt_input, t)
                print("联运:", plan['statistics']['intermodal'], "外包:", plan['statistics']['adhoc'])
                self._apply_plan(plan)

            # 5. 清理新订单池，将订单转移到对应的等待池或完成池
            self._update_pools_after_decision()

            # 6. 计算本周期内完成订单的成本
            self._calculate_costs_for_finished_orders()
            self._capture_snapshot(t, len(batch), 0)

            # 7. 推进时间
            self.state.current_time += self.period_len

        return self._generate_final_results()

    def _generate_final_results(self):
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]

        num_delivered = 0
        num_delayed = 0
        num_adhoc = 0
        total_cost = 0.0

        for o in finished_orders:
            total_cost += (o.total_cost if o.total_cost else 0)
            if o.mode == TransportMode.AD_HOC or o.status == OrderStatus.AD_HOC_MODE:
                num_adhoc += 1
            elif o.status == OrderStatus.DELAYED:
                num_delayed += 1
            elif o.status == OrderStatus.DELIVERED:
                num_delivered += 1

        total_count = len(finished_orders)
        avg_train_util, avg_flight_util = self._calculate_utilization()

        results = {
            'final_metrics': {
                'total_cost': total_cost,
                'num_delivered': num_delivered,
                'num_delayed': num_delayed,
                'num_adhoc': num_adhoc,
                'total_orders': len(self.all_orders),
                'delay_rate': num_delayed / total_count if total_count else 0,
                'adhoc_rate': num_adhoc / total_count if total_count else 0,
                'avg_train_utilization': avg_train_util,
                'avg_flight_utilization': avg_flight_util
            },
            'history': self.history
        }

        self._print_summary()
        return results

    def _calculate_costs_for_finished_orders(self):
        """计算所有已完成但尚未计算成本的订单的总成本"""
        for oid in self.state.pool_finished:
            order = self.state.get_order(oid)

            # 只计算尚未计算成本的订单（避免重复计算）
            if order.total_cost is None or order.total_cost == 0:
                if order.mode == TransportMode.INTERMODAL:
                    # 判断：如果没有分配航班，则视为外包
                    if order.assigned_flight is None:
                        # 部分外包
                        order.total_cost = self.cost_calc.calculate_partial_intermodal_cost(order, order.assigned_train, order.assigned_flight, order.transfer_duration, order.actual_pickup_time)
                    else:
                        # 完整联运
                        order.total_cost = self.cost_calc.calculate_final_intermodal_cost(order, order.assigned_train, order.assigned_flight, order.transfer_duration, order.actual_pickup_time, order.transfer_complete_time, order.actual_flight_time, order.actual_delivery_time)
                elif order.mode == TransportMode.AD_HOC:
                    # 完全外包
                    order.total_cost = self.cost_calc.calculate_direct_adhoc_cost(order)

    def _update_pools_after_decision(self):
        """
        根据 _apply_plan 更新的订单属性，将订单移动到正确的物理池子。
        """
        # 使用切片拷贝遍历，允许在循环中修改原列表
        for oid in list(self.state.pool_new):
            order = self.state.get_order(oid)

            if order.status == OrderStatus.WAITING:
                # 联运订单 -> 移入等待火车池
                self.state.pool_waiting_for_train.add(oid)
                self.state.pool_new.remove(oid)

            elif order.status == OrderStatus.AD_HOC_MODE:
                # 全程外包 -> 直接移入完成池
                self.state.pool_finished.append(oid)
                self.state.pool_new.remove(oid)

    def _get_new_orders(self, current_time, period_len):
        # 简单切片，实际可以用指针优化
        return [o for o in self.all_orders
                if current_time <= o.earliest_pickup < current_time + period_len]

    def _apply_plan(self, plan):
        assignments = plan['order_assignments']

        for oid, decision in assignments.items():
            order = self.state.get_order(oid)
            mode = decision['mode']
            # --- 决策应用逻辑 ---
            if mode == 'INTERMODAL':
                if order.status == OrderStatus.NEW:  # 仅新订单分配资源
                    order.mode = TransportMode.INTERMODAL
                    order.assigned_train = decision['train']
                    order.assigned_flight = decision['flight']  # 可能为 None
                    # 资源扣减
                    self.state.trains[decision['train']].remaining_capacity -= order.volume
                    if order.assigned_flight:  # 只有分配了航班才扣减
                        self.state.flights[decision['flight']].remaining_capacity -= order.volume
                    # 标记状态
                    order.status = OrderStatus.WAITING
                    train = self.state.trains[decision['train']]
                    order.actual_pickup_time = train.get_arrival_at(order.origin)

            elif mode == 'AD_HOC':
                # 场景：直接全程外包
                if order.status == OrderStatus.NEW:
                    order.mode = TransportMode.AD_HOC
                    order.status = OrderStatus.AD_HOC_MODE
                    order.actual_delivery_time = order.deadline  # 假设外包准时

    def _print_summary(self):
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]
        total_cost = sum(o.total_cost for o in finished_orders)
        delayed = [o for o in finished_orders if o.status == OrderStatus.DELAYED]
        print([o.id for o in delayed])
        print(f"\n仿真结束。总成本: {total_cost:.2f}, 延误订单: {len(delayed)}, 完成订单: {len(finished_orders)}, 总订单: {len(self.all_orders)}")


if __name__ == "__main__":
    # 随机种子
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    conf = {
        'period_length': 30,
        'planning_horizon': 15 * 60,   # 900 minutes, matches 7:00–22:00
        'num_scenarios': 50,
        'costs': {}
    }

    # 1. 设置数据目录
    data_dir = '../data/Instance c/'  # 可修改为 './data/Instance a/', './data/Instance b/', './data/Instance c/'

    # 2. 设置具体加载的文件名
    file_params = {
        'train': 'train_50.csv',  # 对应 'train_5.csv', 'train_25.csv', 'train_50.csv'
        'train_station': 'train_with_station_50.csv',  # 对应 'train_with_station_5.csv', 'train_with_station_25.csv', 'train_with_station_50.csv'
        'flight': 'flight_50.csv',  # 对应 'flight_5.csv', 'flight_25.csv', 'flight_50.csv'
        'cargo': 'cargo_1.csv'  # 对应 'cargo_1.csv', 'cargo_2.csv', 'cargo_3.csv', 'cargo_4.csv', 'cargo_5.csv'
    }

    print(f"Running simulation with: {data_dir} | {file_params}")

    sim = Simulator(config=conf, data_dir=data_dir, file_params=file_params)
    results = sim.run()

    # saver = ResultsSaver(base_dir="./simulation_results")
    # saver.save_results(
    #     results,
    #     save_json=True,
    #     save_pickle=True,
    #     save_csv=True,
    #     save_excel=True,
    #     generate_plots=True
    # )
