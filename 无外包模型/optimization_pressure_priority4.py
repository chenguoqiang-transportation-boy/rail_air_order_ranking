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
    两阶段随机规划优化器：
    - 只允许联运（INTERMODAL）
    - 每个订单始终有一个兜底动作：DUMMY（未送达累计到 horizon_time）
    - 若换装中错过航班：不外包，直接转为 DUMMY
    - 决策时在“最优联运(期望成本)”与“DUMMY成本”之间选择更低者（通常联运会更低；若不是请校准成本单位/参数）
    """

    DUMMY_ID = "__DUMMY__"
    TRANSFER_TIME = 75.0

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

        # 必须传 horizon_time
        self.horizon_time = input_data.get("horizon_time")
        if self.horizon_time is None:
            self.horizon_time = 4000

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

    # =========================================================
    # Priority (上帝视角 actual 可行路线)
    # =========================================================
    def _priority_score(self, order: Order, current_time: float, trains: Dict, flights: Dict, indexes: Dict) -> float:
        """
        上帝视角优先级（actual 可行路线）：
        - regret: 再拖一个周期导致的最优成本上升
        - window_urgency: 距离最早可行动窗口（上车/起飞）还有多久
        - scarcity + tightness: 候选少/资源紧
        - slope: 每分钟成本增长率（holding + 若已超期则加罚金）
        """
        t = float(current_time)
        period = getattr(self, "period_len", 30.0)  # 若类里没有 period_len，就默认 30

        routes_now = self._find_feasible_routes_with_actuals(order, trains, flights, indexes)

        if not routes_now:
            return self._unserved_priority_proxy(order, t)

        # 只取前K条低成本路线
        K = 3
        scored_now = [(self._det_cost_actual(order, r), r) for r in routes_now]
        scored_now.sort(key=lambda x: x[0])
        best_cost_now, _ = scored_now[0]
        top_routes = [r for _, r in scored_now[:K]]

        # regret：下一周期最优成本 - 当前最优成本（若下一周期无路，给大值）
        routes_next = self._find_feasible_routes_with_actuals_at_time(order, trains, flights, indexes, t + period)
        if routes_next:
            best_cost_next = min(self._det_cost_actual(order, r) for r in routes_next)
            regret = max(0.0, best_cost_next - best_cost_now)
        else:
            regret = 1e6

        # window urgency：越快到窗口越急
        earliest_pickup = min(r.sch_order_pickup for r in top_routes)
        earliest_flight = min(r.flight_departure_base for r in top_routes)
        time_to_window = min(max(earliest_pickup - t, 0.0), max(earliest_flight - t, 0.0))
        window_urgency = 1.0 / (time_to_window + 1.0)

        scarcity = 1.0 / (len(routes_now) + 1.0)
        tightness = self._resource_tightness(order, top_routes, trains, flights)
        slope = self._marginal_cost_slope(order, t)

        score = (
                0.147897 * regret +
                0.006557947646227 * window_urgency +
                81.46128666762637 * scarcity +
                0.09308689420405693 * tightness +
                312.2970961329239 * slope
        )
        return score

    def _find_feasible_routes_with_actuals_at_time(
        self, order: Order, trains: Dict, flights: Dict, indexes: Dict, t: float
    ) -> List[RouteCandidate]:
        """
        “下一周期再决策”过滤：要求实际取货(上车)时间 >= t
        """
        routes = self._find_feasible_routes_with_actuals(order, trains, flights, indexes)
        return [r for r in routes if r.sch_order_pickup >= t]

    def _find_feasible_routes_with_actuals(
        self, order: Order, trains: Dict, flights: Dict, indexes: Dict
    ) -> List[RouteCandidate]:
        """
        上帝视角：用 actual 时刻判断可行连接
        注意：
        - pickup 推荐用 station.actual_departure（没有则用 actual_arrival）
        - hub 到达推荐用 train.actual_arrival_at_hub（没有则退回 train.actual_arrival）
        """
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
            train_actual_pickup = getattr(origin_station, "actual_departure", origin_station.actual_arrival)
            if train_actual_pickup < order.earliest_pickup:
                continue

            hub_actual_arrival = getattr(train, "actual_arrival_at_hub", train.actual_arrival)

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

    def _det_cost_actual(self, order: Order, route: RouteCandidate) -> float:
        """
        用 actual 时刻计算确定性联运成本（用于priority/打分）
        """
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

    def _resource_tightness(self, order: Order, routes: List[RouteCandidate], trains: Dict, flights: Dict) -> float:
        """
        资源紧张度（0~1）：订单占候选资源剩余容量比例的最大值
        """
        best = 0.0
        for r in routes:
            tr = trains.get(r.train_id)
            fl = flights.get(r.flight_id)
            if tr is None or fl is None:
                continue

            tr_rem = getattr(tr, "remaining_capacity", tr.capacity)
            fl_rem = getattr(fl, "remaining_capacity", fl.capacity)

            train_ratio = order.volume / max(tr_rem, 1e-6)
            flight_ratio = order.volume / max(fl_rem, 1e-6)
            best = max(best, min(2.0, max(train_ratio, flight_ratio)))
        return min(1.0, best)

    def _marginal_cost_slope(self, order: Order, t: float) -> float:
        """
        每分钟成本增长率近似（注意：若你的 holding_per_hour 是“每小时”，这里需要 /60 才是每分钟）
        """
        holding_unit = self.cost_calc.origin_holding_map.get(order.origin, self.cost_calc.costs.holding_per_hour)
        slope = holding_unit * order.volume
        if t >= order.deadline:
            slope += self.cost_calc.costs.delay_penalty_rate * order.volume
        return slope

    def _unserved_priority_proxy(self, order: Order, t: float) -> float:
        slack = max(order.deadline - t, 0.0)
        urgency = 10000.0 / (slack + 1.0)
        slope = self._marginal_cost_slope(order, t)
        return 1e5 * urgency + 1e2 * slope

    # =========================================================
    # Decide (始终可行：联运 vs DUMMY)
    # =========================================================
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
        - 换装中且已确定错过航班：直接 DUMMY
        - 其他情况：在“最优联运期望成本”和“DUMMY成本”之间二选一
        """

        # A) 换装中：若已错过航班 -> DUMMY
        if order.status == OrderStatus.TRANSFERRING:
            if not order.assigned_flight:
                return self._make_dummy_decision(order)

            flight = flights.get(order.assigned_flight)
            if flight is None:
                return self._make_dummy_decision(order)

            if order.transfer_complete_time is not None and order.transfer_complete_time > flight.actual_departure:
                return self._make_dummy_decision(order)

            return None

        # B) 等待类订单：联运 vs DUMMY
        dummy_cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)

        routes = self._find_feasible_routes(order, trains, flights, indexes, current_time)

        best_route = None
        min_expected_cost = float("inf")

        # 计算期望联运成本
        for route in routes:
            exp_cost = self._expected_cost_over_scenarios(order, route, scenarios, dummy_cost=dummy_cost)
            if exp_cost < min_expected_cost:
                min_expected_cost = exp_cost
                best_route = route

        if best_route is None:
            return self._make_decision(TransportMode.INTERMODAL, train=None, flight=None, cost=dummy_cost)

        # 在“最优联运”和“DUMMY”之间选更低成本
        if min_expected_cost <= dummy_cost:
            return self._make_decision(
                mode=TransportMode.INTERMODAL,
                train=best_route.train_id,
                flight=best_route.flight_id,
                cost=min_expected_cost
            )
        else:
            return self._make_decision(TransportMode.INTERMODAL, train=None, flight=None, cost=dummy_cost)

    def _make_dummy_decision(self, order: Order) -> Dict:
        cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)
        return self._make_decision(mode=TransportMode.INTERMODAL, train=None, flight=None, cost=cost)

    # =========================================================
    # Scenario expected cost
    # =========================================================
    def _expected_cost_over_scenarios(
        self,
        order: Order,
        route: RouteCandidate,
        scenarios: List[Scenario],
        dummy_cost: Optional[float] = None
    ) -> float:
        total_cost = 0.0
        if dummy_cost is None:
            dummy_cost = self.cost_calc.calculate_unserved_until_horizon_cost(order, self.horizon_time)

        sch_order_pickup = route.sch_order_pickup
        sch_arrival_at_hub = route.train_arrival_base

        for sc in scenarios:
            accumulated_delays = sc.train_delays.get(route.train_id, {})
            origin_delay = accumulated_delays.get(order.origin, 0.0)
            sim_pickup_time = sch_order_pickup + origin_delay

            hub_delay = list(accumulated_delays.values())[-1] if accumulated_delays else 0.0
            sim_transfer_done = sch_arrival_at_hub + hub_delay + sc.transfer_time

            f_delay = sc.flight_delays.get(route.flight_id, 0.0)
            sim_flight_dep = route.flight_departure_base + f_delay
            sim_flight_arr = sim_flight_dep + route.flight_travel_time

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
                cost = dummy_cost

            total_cost += cost

        return total_cost / max(len(scenarios), 1)

    # =========================================================
    # Feasible routes (scheduled feasibility + remaining capacity)
    # =========================================================
    def _find_feasible_routes(
        self,
        order: Order,
        trains: Dict,
        flights: Dict,
        indexes: Dict,
        current_time: float
    ) -> List[RouteCandidate]:
        """
        “计划视角”可行路线（用于决策阶段）：
        - 用 remaining_capacity
        - 用时刻表 sch_* + 平均换装时间判断
        """
        candidates: List[RouteCandidate] = []

        possible_trains = indexes["train_origin"].get(order.origin, set())
        possible_flights = indexes["flight_dest"].get(order.destination, set())

        transfer_time_mean = self.TRANSFER_TIME

        for tid in possible_trains:
            train = trains.get(tid)
            if not train:
                continue

            # 资源
            if getattr(train, "remaining_capacity", train.capacity) < order.volume:
                continue

            # 时间：列车时刻必须在 earliest_pickup 之后（你原逻辑）
            if train.sch_departure < order.earliest_pickup:
                continue

            station = train.station_map.get(order.origin)
            if station is None:
                continue

            sch_order_pickup = train.sch_departure + station.travel_time_acc

            for fid in possible_flights:
                flight = flights.get(fid)
                if not flight:
                    continue

                if getattr(flight, "remaining_capacity", flight.capacity) < order.volume:
                    continue

                # 航班起飞必须晚于列车到枢纽 + 平均换装
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

    # =========================================================
    # Output helpers
    # =========================================================
    def _make_decision(self, mode: TransportMode, train=None, flight=None, cost: float = 0.0):
        return {"mode": mode.name, "train": train, "flight": flight, "cost": cost}

    def _update_stats(self, stats, decision):
        if decision["mode"] == "INTERMODAL":
            if decision.get("train") is None and decision.get("flight") is None:
                stats["unserved"] += 1
            else:
                stats["intermodal"] += 1