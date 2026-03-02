from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import gurobipy as gp
from gurobipy import GRB
from data_generation import Order, OrderStatus, Scenario, TransportMode
from cost_calculator import CostCalculator


@dataclass
class RouteCandidate:
    train_id: str  # 列车ID
    flight_id: str  # 航班ID
    train_arrival_base: float  # 列车到达时间基准
    flight_departure_base: float  # 航班起飞时间基准
    flight_travel_time: float  # 航班的飞行时间
    sch_order_pickup: float  # 订单预定取货时间


class GlobalOptimizer:
    """
    全局确定性优化器（上帝视角）
    已知所有列车、航班的真实延误情况，对所有订单进行一次性全局最优规划。
    """

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator

    def solve(self, input_data: Dict) -> Dict:
        start_time = time.time()

        # 获取全局数据
        orders: List[Order] = input_data["all_orders"]
        trains = input_data["available_trains"]
        flights = input_data["available_flights"]
        indexes = input_data["indexes"]

        # 注意：这里删除了 scenarios

        assignments = {}
        stats = {"intermodal": 0, "adhoc": 0}

        if orders:
            assignments = self._solve_global_deterministic(
                orders, trains, flights, indexes
            )
            for decision in assignments.values():
                self._update_stats(stats, decision)

        return {
            "order_assignments": assignments,
            "statistics": stats,
            "solve_time": time.time() - start_time,
        }

    def _solve_global_deterministic(
            self,
            orders: List[Order],
            trains: Dict,
            flights: Dict,
            indexes: Dict
    ) -> Dict:
        model = gp.Model("GlobalDeterministicOptimization")
        model.setParam('OutputFlag', 1)  # 全局优化规模大，建议开启日志查看进度
        model.setParam('MIPGap', 0.01)  # 设定1%的间隙以加快求解速度

        order_options = {}

        # 1. 预处理：基于【真实时间】寻找可行路径
        print("正在构建可行路径...")
        for order in orders:
            # 只有两个选择：
            # A. 联运 (Intermodal) - 基于真实时间判断是否赶得上
            # B. 直接外包 (Direct Ad-hoc)

            # 使用真实时间查找路径
            routes = self._find_feasible_routes_with_actuals(order, trains, flights, indexes)

            # 计算外包成本
            adhoc_cost = self.cost_calc.calculate_direct_adhoc_cost(order)

            order_options[order.id] = {
                'routes': routes,
                'adhoc_cost': adhoc_cost
            }

        # 2. 定义变量
        print("正在构建模型变量...")
        x_intermodal = {}
        y_adhoc = {}

        for order in orders:
            opts = order_options[order.id]
            # 外包变量
            y_adhoc[order.id] = model.addVar(vtype=GRB.BINARY, name=f"adhoc_{order.id}")
            # 联运路径变量
            for r_idx in range(len(opts['routes'])):
                x_intermodal[order.id, r_idx] = model.addVar(vtype=GRB.BINARY, name=f"route_{order.id}_{r_idx}")

        # 3. 约束条件
        print("正在添加约束...")

        # (1) 唯一性约束
        for order in orders:
            opts = order_options[order.id]
            route_vars = [x_intermodal[order.id, r_idx] for r_idx in range(len(opts['routes']))]
            model.addConstr(gp.quicksum(route_vars) + y_adhoc[order.id] == 1, name=f"assign_{order.id}")

        # (2) 列车容量约束 (全局所有订单竞争资源)
        for tid, train in trains.items():
            consuming_vars = []
            for order in orders:
                opts = order_options[order.id]
                for r_idx, route in enumerate(opts['routes']):
                    if route.train_id == tid:
                        consuming_vars.append(x_intermodal[order.id, r_idx] * order.volume)
            if consuming_vars:
                # 注意：这里使用 capacity 而不是 remaining_capacity，因为是重新规划一切
                model.addConstr(gp.quicksum(consuming_vars) <= train.capacity, name=f"train_cap_{tid}")

        # (3) 航班容量约束
        for fid, flight in flights.items():
            consuming_vars = []
            for order in orders:
                opts = order_options[order.id]
                for r_idx, route in enumerate(opts['routes']):
                    if route.flight_id == fid:
                        consuming_vars.append(x_intermodal[order.id, r_idx] * order.volume)
            if consuming_vars:
                model.addConstr(gp.quicksum(consuming_vars) <= flight.capacity, name=f"flight_cap_{fid}")

        # 4. 目标函数：最小化确定性总成本
        obj_terms = []
        for order in orders:
            opts = order_options[order.id]

            # 外包成本项
            obj_terms.append(y_adhoc[order.id] * opts['adhoc_cost'])

            # 联运成本项 (直接使用确定性成本)
            for r_idx, route in enumerate(opts['routes']):
                # 这里不需要 scenario 循环，直接计算 perfect cost
                det_cost = self._calculate_deterministic_cost(order, route)
                obj_terms.append(x_intermodal[order.id, r_idx] * det_cost)

        model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

        # 5. 求解
        print("开始求解 MIP...")
        model.optimize()

        # 6. 提取结果
        assignments = {}
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            for order in orders:
                if y_adhoc[order.id].X > 0.5:
                    assignments[order.id] = self._make_decision(
                        TransportMode.AD_HOC, cost=order_options[order.id]['adhoc_cost']
                    )
                else:
                    opts = order_options[order.id]
                    for r_idx, route in enumerate(opts['routes']):
                        if x_intermodal[order.id, r_idx].X > 0.5:
                            assignments[order.id] = self._make_decision(
                                TransportMode.INTERMODAL,
                                route.train_id,
                                route.flight_id,
                                cost=self._calculate_deterministic_cost(order, route)
                            )
                            break
        return assignments

    def _find_feasible_routes_with_actuals(self, order: Order, trains: Dict, flights: Dict, indexes: Dict) -> List[
        RouteCandidate]:
        """
        基于【真实时间(Actual Times)】寻找可行路径。
        这是上帝视角的核心：我们知道如果不延误也赶不上，或者延误了但还是赶得上。
        """
        candidates = []
        possible_trains = indexes["train_origin"].get(order.origin, set())
        possible_flights = indexes["flight_dest"].get(order.destination, set())

        # 换乘时间取固定值或确定的采样值（此处为简化，取均值或直接用DelayModel里的逻辑）
        TRANSFER_TIME = 75.0

        for tid in possible_trains:
            train = trains.get(tid)
            # 容量预筛（可选，主要是为了减少搜索空间）
            if train.capacity < order.volume: continue

            # 【关键修改】：使用 train.actual_departure 判断是否能在订单生成后发车
            # 注意：如果订单生成时间早于发车，就可以赶上
            # 这里需要获取列车在 order.origin 站点的具体发车时间
            station_idx = -1
            for idx, s in enumerate(train.stations):
                if s.origin_id == order.origin:
                    station_idx = idx
                    break

            if station_idx == -1: continue

            origin_station = train.stations[station_idx]
            # 获取该站点的实际离开时间（这需要你在初始化 Simulator 时已经计算好 actual_departure）
            # 假设 train.stations 里存储了 actual_arrival，这里的发车通常近似为 arrival + 停留
            # 简单起见，这里用 actual_arrival 代表该站点的时刻
            train_actual_pickup = origin_station.actual_arrival

            if train_actual_pickup < order.earliest_pickup: continue

            # 获取列车到达枢纽的实际时间
            hub_actual_arrival = train.actual_arrival

            for fid in possible_flights:
                flight = flights.get(fid)
                if flight.capacity < order.volume: continue

                # 【关键修改】：使用实际时间判断换乘连接性
                # 只有当 实际到达枢纽 + 换乘 < 实际起飞 才是可行路径
                if hub_actual_arrival + TRANSFER_TIME <= flight.actual_departure:
                    candidates.append(
                        RouteCandidate(
                            train_id=tid,
                            flight_id=fid,
                            train_arrival_base=hub_actual_arrival,  # 存储实际值
                            flight_departure_base=flight.actual_departure,  # 存储实际值
                            flight_travel_time=flight.travel_time,
                            sch_order_pickup=train_actual_pickup
                        )
                    )
        return candidates

    def _calculate_deterministic_cost(self, order: Order, route: RouteCandidate) -> float:
        """
        计算确定性路径的成本
        """
        # 因为在筛选路径时已经确保了连接性 (actual_arrival + transfer <= actual_departure)
        # 所以这里一定是 FINAL_INTERMODAL 模式

        # 为了调用 calculate_final_intermodal_cost，我们需要组装参数
        # 注意：route 对象里存的已经是 actual 时间了
        actual_pickup = route.sch_order_pickup
        transfer_complete = route.train_arrival_base + 75.0  # 假设换乘固定
        actual_flight_dep = route.flight_departure_base
        actual_delivery = actual_flight_dep + route.flight_travel_time

        return self.cost_calc.calculate_final_intermodal_cost(
            order,
            assigned_train=route.train_id,
            assigned_flight=route.flight_id,
            transfer_duration=75.0,
            actual_pickup_time=actual_pickup,
            transfer_complete_time=transfer_complete,
            actual_flight_time=actual_flight_dep,
            actual_delivery_time=actual_delivery
        )

    def _make_decision(self, mode: TransportMode, train=None, flight=None, cost=0.0):
        return {"mode": mode.name, "train": train, "flight": flight, "cost": cost}

    def _update_stats(self, stats, decision):
        if decision["mode"] == "INTERMODAL":
            stats["intermodal"] += 1
        else:
            stats["adhoc"] += 1

