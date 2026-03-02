from typing import Dict, List
from dataclasses import dataclass
import time

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
        order_options: Dict[str, List[RouteCandidate]] = {}
        for order in orders:
            routes = list(self._find_feasible_routes_with_actuals(order, trains, flights, indexes))
            routes.append(self._make_dummy_route(order))
            order_options[order.id] = routes

        # 类似 optimization_pressure_priority3：先按优先级排序，再逐单分配
        sorted_orders = sorted(
            orders,
            key=lambda o: self._order_cost_priority(o, order_options[o.id]),
            reverse=True,
        )

        remaining_train_capacity = {tid: train.capacity for tid, train in trains.items()}
        remaining_flight_capacity = {fid: flight.capacity for fid, flight in flights.items()}

        assignments = {}
        for order in sorted_orders:
            chosen_route = self._choose_best_route_by_cost(
                order=order,
                routes=order_options[order.id],
                remaining_train_capacity=remaining_train_capacity,
                remaining_flight_capacity=remaining_flight_capacity,
            )
            cost = self._calculate_deterministic_cost(order, chosen_route)

            if chosen_route.train_id == self.DUMMY_ID:
                assignments[order.id] = self._make_decision(
                    TransportMode.INTERMODAL, train=None, flight=None, cost=cost
                )
                continue

            remaining_train_capacity[chosen_route.train_id] -= order.volume
            remaining_flight_capacity[chosen_route.flight_id] -= order.volume
            assignments[order.id] = self._make_decision(
                TransportMode.INTERMODAL,
                train=chosen_route.train_id,
                flight=chosen_route.flight_id,
                cost=cost,
            )

        return assignments

    def _order_cost_priority(self, order: Order, routes: List[RouteCandidate]) -> float:
        """
        成本优先级：订单可行路径中的最低成本（含 DUMMY）。
        数值越大，优先级越高。
        """
        return min(self._calculate_deterministic_cost(order, route) for route in routes)

    def _choose_best_route_by_cost(
        self,
        order: Order,
        routes: List[RouteCandidate],
        remaining_train_capacity: Dict[str, float],
        remaining_flight_capacity: Dict[str, float],
    ) -> RouteCandidate:
        feasible_routes: List[RouteCandidate] = []
        for route in routes:
            if route.train_id == self.DUMMY_ID:
                feasible_routes.append(route)
                continue

            train_remaining = remaining_train_capacity.get(route.train_id, 0.0)
            flight_remaining = remaining_flight_capacity.get(route.flight_id, 0.0)
            if train_remaining >= order.volume and flight_remaining >= order.volume:
                feasible_routes.append(route)

        # 每个订单至少有 DUMMY 兜底
        return min(feasible_routes, key=lambda r: self._calculate_deterministic_cost(order, r))

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
