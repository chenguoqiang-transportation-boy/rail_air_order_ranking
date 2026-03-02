from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from data_generation import Order, OrderStatus, Scenario, TransportMode
from cost_calculator import CostCalculator


@dataclass
class RouteCandidate:
    train_id: str
    flight_id: str
    train_arrival_base: float
    flight_departure_base: float
    flight_travel_time: float
    sch_order_pickup: float


class TwoStageOptimizer:
    """
    两阶段随机规划优化器（改造版：只联运 + 无线路/错过航班 => DUMMY累计到horizon）
    """

    DUMMY_ID = "__DUMMY__"

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator
        self.horizon_time: float = None

    def solve(self, input_data: Dict, current_time: float) -> Dict:
        start_time = time.time()

        orders: List[Order] = input_data["adjustable_orders"]
        trains = input_data["available_trains"]
        flights = input_data["available_flights"]
        indexes = input_data["indexes"]
        scenarios: List[Scenario] = input_data["scenarios"]

        # 关键：传入时间结束（你希望用 max_time）
        self.horizon_time = input_data.get("horizon_time")
        if self.horizon_time is None:
            raise ValueError("TwoStageOptimizer.solve: input_data 必须提供 horizon_time（建议用 max_time）")

        # 优先级排序
        orders = sorted(
            orders,
            key=lambda o: self._priority_score(o, current_time, trains, flights, indexes),
            reverse=True
        )

        assignments = {}
        stats = {"intermodal": 0, "unserved": 0}

        for order in orders:
            decision = self._decide_order(
                order=order,
                trains=trains,
                flights=flights,
                indexes=indexes,
                scenarios=scenarios,
                current_time=current_time,
            )
            if decision:
                assignments[order.id] = decision
                self._update_stats(stats, decision)

        return {
            "order_assignments": assignments,
            "statistics": stats,
            "solve_time": time.time() - start_time,
        }

    # =========================
    # Priority
    # =========================
    def _priority_score(self, order: Order, current_time: float, trains: Dict, flights: Dict, indexes: Dict) -> float:
        deadline = getattr(order, "deadline", None)
        if deadline is None:
            slack = float("inf")
        else:
            slack = max(float(deadline) - float(current_time), 0.0)

        earliest_pickup = getattr(order, "earliest_pickup", 0.0)
        ready_in = max(float(earliest_pickup) - float(current_time), 0.0)

        urgency_term = 10000.0 / (slack + 1.0)
        readiness_term = 100.0 / (ready_in + 1.0)

        # 这里可以加入更多“稀缺性/机会价值”等指标；当前保持你的简化逻辑
        return urgency_term + readiness_term

    # =========================
    # Decide
    # =========================
    def _decide_order(
        self,
        order: Order,
        trains: Dict,
        flights: Dict,
        indexes: Dict,
        scenarios: List[Scenario],
        current_time: float,
    ) -> Optional[Dict]:
        """
        只允许联运：
        - 若可选联运线路存在：选期望成本最低的联运
        - 若无线路：DUMMY（直到 horizon_time 累计成本）
        - 若 TRANSFERRING 且错过航班：不外包，改为 DUMMY（直到 horizon_time）
        """

        # A) 已在换装中：检查是否已经“必然错过航班”
        if order.status == OrderStatus.TRANSFERRING:
            if not order.assigned_flight:
                # 没航班本就不可飞了：按 DUMMY 处理（但通常这种订单应当不会存在于“只联运”体系）
                return self._make_dummy_decision(order)

            flight = flights.get(order.assigned_flight)
            if flight is None:
                return self._make_dummy_decision(order)

            # 若换装完成时间晚于航班实际起飞 -> 必然错过 -> DUMMY
            if order.transfer_complete_time is not None and order.transfer_complete_time > flight.actual_departure:
                return self._make_dummy_decision(order)

            # 能赶上就不调整，继续原计划
            return None

        # B) NEW/WAITING/WAITING_FOR_*：做期望成本最小的联运选择
        routes = self._find_feasible_routes(order, trains, flights, indexes, current_time)
        if not routes:
            return self._make_dummy_decision(order)

        best_route = None
        min_expected_cost = float("inf")

        for route in routes:
            exp_cost = self._expected_cost_over_scenarios(order, route, scenarios)
            if exp_cost < min_expected_cost:
                min_expected_cost = exp_cost
                best_route = route

        if best_route is None:
            return self._make_dummy_decision(order)

        return self._make_decision(
            mode=TransportMode.INTERMODAL,
            train=best_route.train_id,
            flight=best_route.flight_id,
            cost=min_expected_cost
        )

    def _make_dummy_decision(self, order: Order) -> Dict:
        # 统一用 CostCalculator 的 “未送达累计到 horizon” 成本
        cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)
        return self._make_decision(mode=TransportMode.INTERMODAL, train=None, flight=None, cost=cost)

    # =========================
    # Scenario expected cost (联运成功 or 失败=>DUMMY)
    # =========================
    def _expected_cost_over_scenarios(self, order: Order, route: RouteCandidate, scenarios: List[Scenario]) -> float:
        total_cost = 0.0

        sch_order_pickup = route.sch_order_pickup
        sch_arrival_at_hub = route.train_arrival_base

        for sc in scenarios:
            # 1) 场景下“实际轨迹”
            accumulated_delays = sc.train_delays.get(route.train_id, {})
            origin_delay = accumulated_delays.get(order.origin, 0.0)
            sim_pickup_time = sch_order_pickup + origin_delay

            hub_delay = list(accumulated_delays.values())[-1] if accumulated_delays else 0.0
            sim_transfer_done = sch_arrival_at_hub + hub_delay + sc.transfer_time

            f_delay = sc.flight_delays.get(route.flight_id, 0.0)
            sim_flight_dep = route.flight_departure_base + f_delay
            sim_flight_arr = sim_flight_dep + route.flight_travel_time

            # 2) 只考虑两种结果：
            #    - 成功衔接：FINAL_INTERMODAL
            #    - 失败（错过航班）：不外包 -> DUMMY直到horizon
            if sim_transfer_done <= sim_flight_dep:
                cost = self.cost_calc.calculate_final_intermodal_cost(
                    order,
                    assigned_train=route.train_id,
                    assigned_flight=route.flight_id,
                    transfer_duration=sc.transfer_time,
                    actual_pickup_time=sim_pickup_time,
                    transfer_complete_time=sim_transfer_done,
                    actual_flight_time=sim_flight_dep,
                    actual_delivery_time=sim_flight_arr,
                )
            else:
                cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)

            total_cost += cost

        return total_cost / max(len(scenarios), 1)

    # =========================
    # Feasible routes (scheduled feasibility + remaining capacity)
    # =========================
    def _find_feasible_routes(
        self,
        order: Order,
        trains: Dict,
        flights: Dict,
        indexes: Dict,
        current_time: float
    ) -> List[RouteCandidate]:
        candidates: List[RouteCandidate] = []

        possible_trains = indexes["train_origin"].get(order.origin, set())
        possible_flights = indexes["flight_dest"].get(order.destination, set())

        transfer_time_mean = 75.0

        for tid in possible_trains:
            train = trains.get(tid)

            # 检查列车资源
            if not train or train.remaining_capacity < order.volume:
                continue

            # 时间：列车时刻必须在 earliest_pickup 之后（你原逻辑用 sch_departure）
            if train.sch_departure < order.earliest_pickup:
                continue

            station = train.station_map.get(order.origin)
            if station is None:
                continue

            sch_order_pickup = train.sch_departure + station.travel_time_acc

            for fid in possible_flights:
                flight = flights.get(fid)

                # 检查航班资源
                if not flight or flight.remaining_capacity < order.volume:
                    continue

                # 时间：航班时刻必须晚于列车到枢纽 + 平均换装时间
                if flight.sch_departure < train.sch_arrival_at_hub + transfer_time_mean:
                    continue

                candidates.append(
                    RouteCandidate(
                        train_id=tid,
                        flight_id=fid,
                        train_arrival_base=train.sch_arrival_at_hub,
                        flight_departure_base=flight.sch_departure,
                        flight_travel_time=flight.travel_time,
                        sch_order_pickup=sch_order_pickup,
                    )
                )

        return candidates

    # =========================
    # Output helpers
    # =========================
    def _make_decision(self, mode: TransportMode, train=None, flight=None, cost: float = 0.0):
        return {"mode": mode.name, "train": train, "flight": flight, "cost": cost}

    def _update_stats(self, stats, decision):
        # INTERMODAL 下再细分 unserved
        if decision["mode"] == "INTERMODAL":
            if decision.get("train") is None and decision.get("flight") is None:
                stats["unserved"] += 1
            else:
                stats["intermodal"] += 1