import random
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict

from data_generation import (
    DataLoader, NetworkInitializer, OrderGenerator, ScenarioGenerator,
    DelayModel, Costs, Order, TransportMode, OrderStatus
)
from cost_calculator import CostCalculator
from optimization_pressure_priority4 import TwoStageOptimizer


class SimulationState:
    """存储仿真的所有状态数据，管理订单在不同阶段的池子"""
    def __init__(self):
        self.current_time = 0.0
        self.orders: Dict[str, Order] = {}
        self.trains = {}
        self.flights = {}

        # 订单集合 (使用 ID)
        self.pool_new: List[str] = []                 # 新到达订单
        self.pool_waiting_for_train: set = set()      # 等待上火车
        self.pool_on_train: set = set()               # 在火车上
        self.pool_transferring: set = set()           # 换装中
        self.pool_waiting_for_flight: set = set()     # 等待登机
        self.pool_on_flight: set = set()              # 在飞机上

        # 【新增】无联运线路 / DUMMY订单池：不走运输，到 horizon_time 才结算完成
        self.pool_unserved: set = set()

        self.pool_finished: List[str] = []            # 已完成

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
    """负责物理世界的状态演进，按时间顺序处理（只联运 + DUMMY结算）"""
    def __init__(self, state: SimulationState, horizon_time: float):
        self.state = state
        self.horizon_time = horizon_time

    def process_events(self):
        t = self.state.current_time

        # ==========================================
        # 0) DUMMY订单：不参与物理运输，只在 horizon_time 时结算为完成
        # ==========================================
        if t >= self.horizon_time:
            for oid in list(self.state.pool_unserved):
                order = self.state.get_order(oid)
                order.actual_delivery_time = self.horizon_time
                order.status = OrderStatus.DELAYED if self.horizon_time > order.deadline else OrderStatus.DELIVERED
                self.state.pool_unserved.remove(oid)
                self.state.pool_finished.append(oid)

        # ==========================================
        # 1) 上车逻辑(WAITING -> ON_TRAIN)
        # ==========================================
        for oid in list(self.state.pool_waiting_for_train):
            order = self.state.get_order(oid)
            if order.assigned_train and order.actual_pickup_time is not None and t >= order.actual_pickup_time:
                order.status = OrderStatus.ON_TRAIN
                self.state.pool_waiting_for_train.remove(oid)
                self.state.pool_on_train.add(oid)

        # ==========================================
        # 2) 列车到达 (ON_TRAIN -> TRANSFERRING)
        # ==========================================
        for oid in list(self.state.pool_on_train):
            order = self.state.get_order(oid)
            train = self.state.trains[order.assigned_train]
            if t >= train.actual_arrival:
                order.status = OrderStatus.TRANSFERRING
                order.actual_transfer_start = train.actual_arrival
                order.transfer_duration = DelayModel.sample_transfer_time()
                order.transfer_complete_time = order.actual_transfer_start + order.transfer_duration

                self.state.pool_on_train.remove(oid)
                self.state.pool_transferring.add(oid)

        # ==========================================
        # 3) 换装完成 (TRANSFERRING -> WAITING_FOR_FLIGHT)
        #    【重要】不再存在“无航班=外包”分支
        # ==========================================
        for oid in list(self.state.pool_transferring):
            order = self.state.get_order(oid)
            if order.transfer_complete_time is not None and t >= order.transfer_complete_time:
                self.state.pool_transferring.remove(oid)
                order.status = OrderStatus.WAITING_FOR_FLIGHT
                self.state.pool_waiting_for_flight.add(oid)

        # ==========================================
        # 4) 登机 (WAITING_FOR_FLIGHT -> ON_FLIGHT)
        #    【补全】写入 actual_flight_time
        # ==========================================
        for oid in list(self.state.pool_waiting_for_flight):
            order = self.state.get_order(oid)
            flight = self.state.flights.get(order.assigned_flight)
            if flight and t >= flight.actual_departure:
                order.status = OrderStatus.ON_FLIGHT
                order.actual_flight_time = flight.actual_departure
                self.state.pool_waiting_for_flight.remove(oid)
                self.state.pool_on_flight.add(oid)

        # ==========================================
        # 5) 航班到达 (ON_FLIGHT -> DELIVERED/DELAYED)
        # ==========================================
        for oid in list(self.state.pool_on_flight):
            order = self.state.get_order(oid)
            flight = self.state.flights.get(order.assigned_flight)
            if flight and t >= flight.actual_arrival:
                order.actual_delivery_time = flight.actual_arrival
                order.status = OrderStatus.DELAYED if order.actual_delivery_time > order.deadline else OrderStatus.DELIVERED
                self.state.pool_on_flight.remove(oid)
                self.state.pool_finished.append(oid)


class Simulator:
    def __init__(self, config: Dict, data_dir: str, file_params: Dict[str, str]):
        self.config = config
        self.costs = Costs(**config.get('costs', {}))

        self.period_len = config.get('period_length', 30)
        self.horizon = config.get('planning_horizon', 900)

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
        self.scenario_gen = ScenarioGenerator(DelayModel())

        self.all_orders = OrderGenerator(self.data['cargo']).generate()
        self.all_orders.sort(key=lambda x: x.earliest_pickup)

        # 你要求：时间结束用 max_time
        self.max_time = self.all_orders[-1].deadline + 2000

        # 引擎：用 max_time 结算 DUMMY
        self.engine = TransportationEngine(self.state, horizon_time=self.max_time)

        self.history = []

    def _init_network(self):
        """初始化并注入延误（上帝视角）"""
        self.state.trains = NetworkInitializer.init_trains(
            self.data['train'], self.data['train_station'], self.data['origin']
        )
        self.state.flights = NetworkInitializer.init_flights(
            self.data['flight'], self.data['destination']
        )
        self._build_static_indexes()

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
                if i == len(t.stations) - 1:
                    t.actual_arrival = s.actual_arrival

        for f in self.state.flights.values():
            d = dm.sample_flight_delay(f.average_delay, f.worst_delay)
            f.actual_departure = f.sch_departure + d
            f.actual_arrival = f.actual_departure + f.travel_time

    def _build_static_indexes(self):
        for t_id, train in self.state.trains.items():
            for s in train.stations:
                self.state.train_origin_idx[s.origin_id].add(t_id)

        for f_id, flight in self.state.flights.items():
            self.state.flight_dest_idx[flight.destination_id].add(f_id)

    def _calculate_utilization(self):
        train_utils = []
        for t in self.state.trains.values():
            if t.capacity > 0:
                util = (t.capacity - t.remaining_capacity) / t.capacity
                train_utils.append(util)
        avg_train = sum(train_utils) / len(train_utils) if train_utils else 0

        flight_utils = []
        for f in self.state.flights.values():
            if f.capacity > 0:
                util = (f.capacity - f.remaining_capacity) / f.capacity
                flight_utils.append(util)
        avg_flight = sum(flight_utils) / len(flight_utils) if flight_utils else 0

        return avg_train, avg_flight

    def _capture_snapshot(self, t, new_orders_count, adjustable_count):
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]

        num_delivered = 0
        num_delayed = 0
        num_unserved = 0
        total_cost = 0.0

        for o in finished_orders:
            total_cost += (o.total_cost if o.total_cost else 0)
            if o.mode == TransportMode.INTERMODAL and o.assigned_train is None and o.assigned_flight is None:
                num_unserved += 1
            elif o.status == OrderStatus.DELAYED:
                num_delayed += 1
            elif o.status == OrderStatus.DELIVERED:
                num_delivered += 1

        total_finished = len(finished_orders)
        delay_rate = num_delayed / total_finished if total_finished > 0 else 0
        unserved_rate = num_unserved / total_finished if total_finished > 0 else 0

        snapshot = {
            'time': t,
            'new_orders': new_orders_count,
            'adjustable_orders': adjustable_count,
            'frozen_train': len(self.state.pool_on_train) + len(self.state.pool_transferring),
            'frozen_flight': len(self.state.pool_on_flight),
            'metrics': {
                'total_cost': total_cost,
                'num_delivered': num_delivered,
                'num_delayed': num_delayed,
                'num_unserved': num_unserved,
                'delay_rate': delay_rate,
                'unserved_rate': unserved_rate
            }
        }
        self.history.append(snapshot)

    def run(self):
        print(f"开始仿真: {len(self.all_orders)} 订单, 周期 {self.period_len}min")
        self.history = []

        while self.state.current_time < self.max_time:
            t = self.state.current_time
            print("=================当前时刻:", int(t), "s=================")

            # 1) 物理状态演进（含 DUMMY 在 max_time 结算）
            self.engine.process_events()

            # 2) 新订单到达
            batch = self._get_new_orders(t, self.period_len)
            self.state.add_new_orders(batch)

            # 3) 需要决策的订单：新订单 + 等待类订单 +（可选）换装中订单
            adjustable_orders = (
                [self.state.get_order(oid) for oid in self.state.pool_new] +
                [self.state.get_order(oid) for oid in self.state.pool_waiting_for_train] +
                [self.state.get_order(oid) for oid in self.state.pool_waiting_for_flight] +
                [self.state.get_order(oid) for oid in self.state.pool_transferring]   # 允许TwoStage对“错过航班”做DUMMY调整
            )

            if adjustable_orders:
                # 可用资源窗口（保持你的原筛选）
                avail_trains = {k: v for k, v in self.state.trains.items() if t <= v.sch_departure <= t + self.horizon}
                avail_flights = {k: v for k, v in self.state.flights.items() if t <= v.sch_departure <= t + self.horizon}

                # 用于场景生成的 Train Actuals
                train_actuals = {
                    tid: ([s.station_delay for s in tr.stations],
                          [s.actual_arrival for s in tr.stations])
                    for tid, tr in avail_trains.items()
                }

                opt_input = {
                    'adjustable_orders': adjustable_orders,
                    'available_trains': avail_trains,
                    'available_flights': avail_flights,
                    'indexes': {
                        'train_origin': self.state.train_origin_idx,
                        'flight_dest': self.state.flight_dest_idx
                    },
                    'scenarios': self.scenario_gen.generate(
                        avail_trains, avail_flights,
                        self.config.get('num_scenarios', 50),
                        t, train_actuals
                    ),
                    # 【关键】传入时间结束，用 max_time
                    'horizon_time': self.max_time
                }

                # 4) 执行优化（TwoStageOptimizer 已改为只联运/或DUMMY）
                plan = self.optimizer.solve(opt_input, t)
                intermodal = plan['statistics'].get('intermodal', 0)
                unserved = plan['statistics'].get('unserved', 0)
                print("联运:", intermodal, "无线路(DUMMY):", unserved)

                self._apply_plan(plan)

            # 5) pool_new -> waiting / unserved
            self._update_pools_after_decision()

            # 6) 计算完成订单成本 + 快照
            self._calculate_costs_for_finished_orders()
            self._capture_snapshot(t, len(batch), len(adjustable_orders))

            # 7) 推进时间
            self.state.current_time += self.period_len

        # ====== 关键补丁：补一个最终 tick，让 t==max_time 时结算 DUMMY ======
        self.state.current_time = self.max_time
        self.engine.process_events()
        self._calculate_costs_for_finished_orders()
        self._capture_snapshot(self.state.current_time, 0, 0)

        return self._generate_final_results()

    def _generate_final_results(self):
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]

        num_delivered = 0
        num_delayed = 0
        num_unserved = 0
        total_cost = 0.0

        for o in finished_orders:
            total_cost += (o.total_cost if o.total_cost else 0)
            if o.mode == TransportMode.INTERMODAL and o.assigned_train is None and o.assigned_flight is None:
                num_unserved += 1
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
                'num_unserved': num_unserved,
                'total_orders': len(self.all_orders),
                'delay_rate': num_delayed / total_count if total_count else 0,
                'unserved_rate': num_unserved / total_count if total_count else 0,
                'avg_train_utilization': avg_train_util,
                'avg_flight_utilization': avg_flight_util
            },
            'history': self.history
        }

        self._print_summary()
        return results

    def _calculate_costs_for_finished_orders(self):
        """只计算：正常联运成本 or DUMMY累计到max_time的成本"""
        for oid in self.state.pool_finished:
            order = self.state.get_order(oid)

            if order.total_cost is None or order.total_cost == 0:
                if order.mode == TransportMode.INTERMODAL:
                    # DUMMY：未送达到 max_time
                    if order.assigned_train is None and order.assigned_flight is None:
                        order.total_cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.max_time)
                    else:
                        order.total_cost = self.cost_calc.calculate_final_intermodal_cost(
                            order,
                            order.assigned_train,
                            order.assigned_flight,
                            order.transfer_duration,
                            order.actual_pickup_time,
                            order.transfer_complete_time,
                            order.actual_flight_time,
                            order.actual_delivery_time
                        )

    def _update_pools_after_decision(self):
        """根据决策把 pool_new 订单移入 waiting_for_train 或 unserved"""
        for oid in list(self.state.pool_new):
            order = self.state.get_order(oid)

            # DUMMY：进入 unserved，不进入等待上车
            if order.mode == TransportMode.INTERMODAL and order.assigned_train is None and order.assigned_flight is None:
                self.state.pool_new.remove(oid)
                self.state.pool_unserved.add(oid)
                continue

            if order.status == OrderStatus.WAITING:
                self.state.pool_waiting_for_train.add(oid)
                self.state.pool_new.remove(oid)

    def _get_new_orders(self, current_time, period_len):
        return [o for o in self.all_orders if current_time <= o.earliest_pickup < current_time + period_len]

    def _apply_plan(self, plan):
        """
        TwoStageOptimizer 已保证只返回 INTERMODAL；
        其中 train=None & flight=None 表示 DUMMY。
        """
        assignments = plan['order_assignments']

        for oid, decision in assignments.items():
            order = self.state.get_order(oid)
            if order is None:
                continue

            if decision['mode'] != 'INTERMODAL':
                # 理论上不会出现，兜底忽略
                continue

            # 只有“可决策阶段”的订单才覆盖（避免把已冻结订单强行改计划）
            if order.status not in (
                OrderStatus.NEW,
                OrderStatus.WAITING,
                OrderStatus.WAITING_FOR_FLIGHT,
                OrderStatus.TRANSFERRING
            ):
                continue

            order.mode = TransportMode.INTERMODAL
            order.assigned_train = decision.get('train')
            order.assigned_flight = decision.get('flight')

            # DUMMY：不扣资源，等待 max_time 结算
            if order.assigned_train is None and order.assigned_flight is None:
                order.status = OrderStatus.WAITING
                order.actual_delivery_time = self.max_time
                continue

            # 正常联运：扣资源（对 remaining_capacity 的扣减）
            if order.assigned_train:
                self.state.trains[order.assigned_train].remaining_capacity -= order.volume
            if order.assigned_flight:
                self.state.flights[order.assigned_flight].remaining_capacity -= order.volume

            # 标记状态 + 设置上车时间（用站点 actual_arrival 近似）
            order.status = OrderStatus.WAITING
            train = self.state.trains.get(order.assigned_train)
            if train:
                # 你原来用 train.get_arrival_at(order.origin)，这里用 station_map 更通用
                st = train.station_map.get(order.origin)
                if st and hasattr(st, "actual_arrival"):
                    order.actual_pickup_time = st.actual_arrival
                else:
                    # 兜底：若没有 actual_arrival，退回用 sch
                    order.actual_pickup_time = getattr(train, "sch_departure", order.earliest_pickup)

    def _print_summary(self):
        finished_orders = [self.state.get_order(oid) for oid in self.state.pool_finished]
        total_cost = sum((o.total_cost or 0.0) for o in finished_orders)
        delayed = [o for o in finished_orders if o.status == OrderStatus.DELAYED]
        print([o.id for o in delayed])
        print(
            f"\n仿真结束。总成本: {total_cost:.2f}, 延误订单: {len(delayed)}, "
            f"完成订单: {len(finished_orders)}, 总订单: {len(self.all_orders)}"
        )


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    conf = {
        'period_length': 30,
        'planning_horizon': 15 * 60,
        'num_scenarios': 50,
        'costs': {}
    }

    data_dir = '../data/Instance c/'
    file_params = {
        'train': 'train_50.csv',
        'train_station': 'train_with_station_50.csv',
        'flight': 'flight_50.csv',
        'cargo': 'cargo_1.csv'
    }

    print(f"Running simulation with: {data_dir} | {file_params}")

    sim = Simulator(config=conf, data_dir=data_dir, file_params=file_params)
    results = sim.run()