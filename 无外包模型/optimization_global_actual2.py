from typing import Dict, List
from dataclasses import dataclass
import time
import gurobipy as gp
from gurobipy import GRB

from data_generation import Order, TransportMode
from cost_calculator import CostCalculator


@dataclass
class RouteCandidate:
    train_id: str
    flight_id: str
    train_arrival_base: float
    flight_departure_base: float
    flight_travel_time: float
    sch_order_pickup: float


class GlobalOptimizer:
    """
    全局确定性优化器（上帝视角）
    - 只允许联运（INTERMODAL）
    - 若无任何联运可行线路，则使用 DUMMY 路径（不消耗容量）
    - DUMMY 成本：一直累计到 horizon_time（这里用 max_time 传入）
    """

    DUMMY_ID = "__DUMMY__"
    TRANSFER_TIME = 75.0

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator
        self.horizon_time: float = None

    def solve(self, input_data: Dict) -> Dict:
        start_time = time.time()

        orders: List[Order] = input_data["all_orders"]
        trains = input_data["available_trains"]
        flights = input_data["available_flights"]
        indexes = input_data["indexes"]

        self.horizon_time = input_data.get("horizon_time")
        if self.horizon_time is None:
            raise ValueError("GlobalOptimizer.solve: input_data 必须提供 horizon_time（这里建议用 max_time）")

        assignments = {}
        stats = {"intermodal": 0, "unserved": 0}

        if orders:
            assignments = self._solve_global_deterministic(orders, trains, flights, indexes)
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
        model.setParam("OutputFlag", 1)
        model.setParam("MIPGap", 0.01)

        order_options: Dict[str, Dict[str, List[RouteCandidate]]] = {}

        # --- sanity: order ids ---
        ids = [o.id for o in orders]
        print("[DEBUG] unique order ids:", len(set(ids)), "total:", len(ids))

        # 1) 预处理：找可行联运；若空则补 DUMMY
        print("正在构建可行路径...")
        for order in orders:
            routes = self._find_feasible_routes_with_actuals(order, trains, flights, indexes)

            # 无论是否有联运，都加入 DUMMY 作为“不可服务/延误到horizon”的兜底路径
            routes = list(routes)  # 防止返回的是生成器/tuple
            routes.append(self._make_dummy_route(order))

            order_options[order.id] = {"routes": routes}

        dummy_only = 0
        for o in orders:
            rs = order_options[o.id]["routes"]
            has_real = any(r.train_id != self.DUMMY_ID for r in rs)
            if not has_real:
                dummy_only += 1
        print("[DEBUG] dummy_only(no intermodal candidates):", dummy_only)

        # 2) 变量（建完后必须 update，再看 NumVars 才准）
        print("正在构建模型变量...")
        x_intermodal = {}
        var_count = 0

        for order in orders:
            routes = order_options[order.id]["routes"]
            if routes is None or len(routes) == 0:
                # 强制补 dummy
                order_options[order.id]["routes"] = [self._make_dummy_route(order)]
                routes = order_options[order.id]["routes"]

            for r_idx in range(len(routes)):
                x_intermodal[order.id, r_idx] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"route_{order.id}_{r_idx}"
                )
                var_count += 1

        model.update()
        print("[DEBUG] after vars: var_count=", var_count, "NumVars=", model.NumVars, "NumConstrs=", model.NumConstrs)
        if model.NumVars == 0:
            raise RuntimeError("No variables were added to the model. Check the var-building loop and module you are running.")

        # 3) 约束
        print("正在添加约束...")

        # (1) 唯一性：每订单必须选 exactly one route
        for order in orders:
            routes = order_options[order.id]["routes"]
            if routes is None or len(routes) == 0:
                order_options[order.id]["routes"] = [self._make_dummy_route(order)]
                routes = order_options[order.id]["routes"]
                if (order.id, 0) not in x_intermodal:
                    x_intermodal[order.id, 0] = model.addVar(vtype=GRB.BINARY, name=f"route_{order.id}_0")

            # 用 LinExpr 显式构造（排除 quicksum 异常情况）
            lhs = gp.LinExpr()
            for r_idx in range(len(routes)):
                lhs += x_intermodal[order.id, r_idx]
            model.addConstr(lhs == 1.0, name=f"assign_{order.id}")

        model.update()
        print("[DEBUG] after assign: NumVars=", model.NumVars, "NumConstrs=", model.NumConstrs)

        # (2) 列车容量：DUMMY 不消耗
        for tid, train in trains.items():
            consuming = gp.LinExpr()
            used = False
            for order in orders:
                for r_idx, route in enumerate(order_options[order.id]["routes"]):
                    if route.train_id == self.DUMMY_ID:
                        continue
                    if route.train_id == tid:
                        consuming += x_intermodal[order.id, r_idx] * order.volume
                        used = True
            if used:
                model.addConstr(consuming <= train.capacity, name=f"train_cap_{tid}")

        model.update()
        print("[DEBUG] after train caps: NumConstrs=", model.NumConstrs)

        # (3) 航班容量：DUMMY 不消耗
        for fid, flight in flights.items():
            consuming = gp.LinExpr()
            used = False
            for order in orders:
                for r_idx, route in enumerate(order_options[order.id]["routes"]):
                    if route.flight_id == self.DUMMY_ID:
                        continue
                    if route.flight_id == fid:
                        consuming += x_intermodal[order.id, r_idx] * order.volume
                        used = True
            if used:
                model.addConstr(consuming <= flight.capacity, name=f"flight_cap_{fid}")

        model.update()
        print("[DEBUG] after flight caps: NumConstrs=", model.NumConstrs)

        # --- DEBUG: objective coefficient sanity check ---
        zero_cnt = 0
        nan_cnt = 0
        neg_cnt = 0
        pos_cnt = 0
        min_c = float("inf")
        max_c = -float("inf")
        sample_print = 0

        for order in orders:
            for route in order_options[order.id]["routes"]:
                c = self._calculate_deterministic_cost(order, route)
                if c is None:
                    nan_cnt += 1
                    continue
                if isinstance(c, float) and (c != c):  # NaN
                    nan_cnt += 1
                    continue
                if c == 0:
                    zero_cnt += 1
                elif c > 0:
                    pos_cnt += 1
                else:
                    neg_cnt += 1
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                if sample_print < 10:
                    print("[DEBUG cost sample]", order.id, route.train_id, route.flight_id, c)
                    sample_print += 1

        print("[DEBUG cost stats] min:", min_c, "max:", max_c,
              "zero:", zero_cnt, "pos:", pos_cnt, "neg:", neg_cnt, "nan:", nan_cnt)

        # 4) 目标：最小化确定性成本（联运 or DUMMY）
        obj = gp.LinExpr()
        for order in orders:
            for r_idx, route in enumerate(order_options[order.id]["routes"]):
                det_cost = self._calculate_deterministic_cost(order, route)
                obj += x_intermodal[order.id, r_idx] * det_cost

        model.setObjective(obj, GRB.MINIMIZE)

        model.update()
        print("[DEBUG] before optimize: NumVars=", model.NumVars, "NumConstrs=", model.NumConstrs)

        # 5) 求解
        print("开始求解 MIP...")
        model.optimize()

        # 6) 提取结果
        assignments = {}

        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            for order in orders:
                routes = order_options[order.id]["routes"]
                chosen_idx = None
                for r_idx in range(len(routes)):
                    if x_intermodal[order.id, r_idx].X > 0.5:
                        chosen_idx = r_idx
                        break

                if chosen_idx is None:
                    # 这里不要悄悄兜底成 DUMMY，否则会掩盖模型问题
                    raise RuntimeError(
                        f"No route selected for order {order.id}. "
                        f"Check that assign constraints exist and the model is built correctly."
                    )

                route = routes[chosen_idx]
                cost = self._calculate_deterministic_cost(order, route)
                if route.train_id == self.DUMMY_ID:
                    assignments[order.id] = self._make_decision(
                        TransportMode.INTERMODAL, train=None, flight=None, cost=cost
                    )
                else:
                    assignments[order.id] = self._make_decision(
                        TransportMode.INTERMODAL, train=route.train_id, flight=route.flight_id, cost=cost
                    )

        else:
            print(f"[WARN] Global MIP status={model.status}. Fallback: all orders -> DUMMY until horizon.")
            for order in orders:
                assignments[order.id] = self._make_decision(
                    TransportMode.INTERMODAL, train=None, flight=None,
                    cost=self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)
                )

        return assignments

    def _make_dummy_route(self, order: Order) -> RouteCandidate:
        return RouteCandidate(
            train_id=self.DUMMY_ID,
            flight_id=self.DUMMY_ID,
            train_arrival_base=0.0,
            flight_departure_base=0.0,
            flight_travel_time=0.0,
            sch_order_pickup=order.earliest_pickup
        )

    def _find_feasible_routes_with_actuals(
        self, order: Order, trains: Dict, flights: Dict, indexes: Dict
    ) -> List[RouteCandidate]:
        candidates: List[RouteCandidate] = []
        possible_trains = indexes["train_origin"].get(order.origin, set())
        possible_flights = indexes["flight_dest"].get(order.destination, set())

        for tid in possible_trains:
            train = trains.get(tid)
            if train is None or train.capacity < order.volume:
                continue

            station_idx = -1
            for idx, s in enumerate(train.stations):
                if s.origin_id == order.origin:
                    station_idx = idx
                    break
            if station_idx == -1:
                continue

            origin_station = train.stations[station_idx]
            # 如果你有 actual_departure，优先用它；否则仍用 actual_arrival
            train_actual_pickup = getattr(origin_station, "actual_departure", origin_station.actual_arrival)
            if train_actual_pickup < order.earliest_pickup:
                continue

            # 注意：这里用的是 train.actual_arrival（若不是hub到达时间，会错杀路线）
            hub_actual_arrival = train.actual_arrival

            for fid in possible_flights:
                flight = flights.get(fid)
                if flight is None or flight.capacity < order.volume:
                    continue

                if hub_actual_arrival + self.TRANSFER_TIME <= flight.actual_departure:
                    candidates.append(
                        RouteCandidate(
                            train_id=tid,
                            flight_id=fid,
                            train_arrival_base=hub_actual_arrival,
                            flight_departure_base=flight.actual_departure,
                            flight_travel_time=flight.travel_time,
                            sch_order_pickup=train_actual_pickup
                        )
                    )

        return candidates

    def _calculate_deterministic_cost(self, order: Order, route: RouteCandidate) -> float:
        if route.train_id == self.DUMMY_ID:
            return self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)

        actual_pickup = route.sch_order_pickup
        transfer_complete = route.train_arrival_base + self.TRANSFER_TIME
        actual_flight_dep = route.flight_departure_base
        actual_delivery = actual_flight_dep + route.flight_travel_time

        return self.cost_calc.calculate_final_intermodal_cost(
            order,
            assigned_train=route.train_id,
            assigned_flight=route.flight_id,
            transfer_duration=self.TRANSFER_TIME,
            actual_pickup_time=actual_pickup,
            transfer_complete_time=transfer_complete,
            actual_flight_time=actual_flight_dep,
            actual_delivery_time=actual_delivery
        )

    def _make_decision(self, mode: TransportMode, train=None, flight=None, cost: float = 0.0):
        return {"mode": mode.name, "train": train, "flight": flight, "cost": cost}

    def _update_stats(self, stats, decision):
        if decision["mode"] == "INTERMODAL":
            if decision["train"] is None and decision["flight"] is None:
                stats["unserved"] += 1
            else:
                stats["intermodal"] += 1