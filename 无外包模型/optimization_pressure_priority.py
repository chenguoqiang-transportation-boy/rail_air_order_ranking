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
    两阶段随机规划优化器（重构版）
    - 统一订单入口：orders
    - 统一决策入口：_decide_order
    - 统一三情形成本计算：_calc_cost_for_case
    """

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator

    def solve(self, input_data: Dict, current_time: float) -> Dict:
        start_time = time.time()

        orders: List[Order] = input_data["adjustable_orders"]
        trains = input_data["available_trains"]
        flights = input_data["available_flights"]
        indexes = input_data["indexes"]
        scenarios: List[Scenario] = input_data["scenarios"]

        # 基于“优先级分数”排序：分数越高越先决策（不再单独处理同分的次序）
        orders = sorted(orders, key=lambda o: self._priority_score(o, current_time, trains, flights, indexes), reverse=True)

        assignments = {}
        stats = {"intermodal": 0, "adhoc": 0}

        # 统一处理：根据订单状态在 _decide_order 内部选择“场景期望”或“实际观测”
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



    def _priority_score(self, order: Order, current_time: float, trains: Dict, flights: Dict, indexes: Dict) -> float:
        """计算订单的优先级分数（分数越大越优先）。

        基础项（局部紧迫度）：
        - slack = max(deadline - current_time, 0)：离截止越近越优先
        - ready_in = max(earliest_pickup - current_time, 0)：越快可开始越优先

        全局感知项（组合优化友好）：
        - 机会价值：该订单“联运 vs 直接外包”的潜在节省（越大越优先）
        - 稀缺性：可行联运候选的容量余量越紧（越容易被后续订单挤掉）越优先
        - 灵活性：可行候选越少（越容易错过组合机会）越优先

        设计目标：在不做完整组合优化的前提下，通过“机会价值 + 稀缺性 + 灵活性”
        让排序具备全局意识，减少贪心逐单决策导致的联运机会流失。
        """
        # ---------- 1) 局部紧迫度：slack / ready_in ----------
        deadline = getattr(order, "deadline", None)
        if deadline is None:
            slack = float("inf")
        else:
            slack = max(float(deadline) - float(current_time), 0.0)

        earliest_pickup = getattr(order, "earliest_pickup", 0.0)
        ready_in = max(float(earliest_pickup) - float(current_time), 0.0)

        # 倒数形式：时间越小 -> 分数越大
        urgency_term = 10000.0 / (slack + 1.0)
        readiness_term = 100.0 / (ready_in + 1.0)

        # ---------- 2) 全局感知：机会价值 / 稀缺性 / 灵活性 ----------
        candidates = self._find_feasible_routes(order, trains, flights, indexes, current_time)
        if not candidates:
            return urgency_term + readiness_term

        # 2.1 机会价值：联运相对直接外包的潜在节省（取最大节省的候选，作为“最好能组合”的信号）
        try:
            adhoc_cost = self._calc_cost_for_case("DIRECT_ADHOC", order)
        except Exception:
            # 极端情况下（成本计算缺字段等），保守回退
            adhoc_cost = 0.0

        transfer_time_mean = 75.0
        best_saving = 0.0

        # 2.2 稀缺性：候选里“最紧”的容量余量（越小越稀缺，越需要尽早占位）
        # margin = min(train_cap, flight_cap) / volume，margin 越小越稀缺
        min_margin = float("inf")

        for c in candidates:
            train = trains.get(c.train_id)
            flight = flights.get(c.flight_id)
            if not train or not flight:
                continue

            # 容量余量（相对订单体量）
            try:
                margin = min(train.remaining_capacity, flight.remaining_capacity) / max(float(order.volume), 1e-9)
            except Exception:
                margin = float("inf")
            if margin < min_margin:
                min_margin = margin

            # 用“基准时刻 + 均值换装”构造一个快速、可比的联运成本近似
            try:
                intermodal_cost = self._calc_cost_for_case(
                    "FINAL_INTERMODAL",
                    order,
                    train_id=c.train_id,
                    flight_id=c.flight_id,
                    transfer_time=transfer_time_mean,
                    actual_pickup_time=c.sch_order_pickup,
                    transfer_complete_time=c.train_arrival_base + transfer_time_mean,
                    actual_flight_time=c.flight_travel_time,
                    actual_delivery_time=c.flight_departure_base + c.flight_travel_time,
                )
                saving = adhoc_cost - intermodal_cost
                if saving > best_saving:
                    best_saving = saving
            except Exception:
                # 任何字段缺失/成本函数内部异常，都不让排序崩掉
                pass

        # 2.3 灵活性：候选越少越优先（避免后续订单把为数不多的组合机会占掉）
        flexibility_term = 1.0 / max(len(candidates), 1)

        # 稀缺性项：margin 越小 -> 1/(margin+1) 越大；若无法计算则置 0
        if min_margin == float("inf"):
            scarcity_term = 0.0
        else:
            scarcity_term = 1.0 / (min_margin + 1.0)

        # 权重：给全局项一个“温和但可见”的影响，避免压过硬约束的紧迫度
        W_SAVING = 1.0
        W_SCARCITY = 500.0
        W_FLEX = 200.0

        global_term = W_SAVING * max(best_saving, 0.0) + W_SCARCITY * scarcity_term + W_FLEX * flexibility_term

        return urgency_term + readiness_term + 100*global_term

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
        根据订单状态决定策略（先处理“已经发生的事实”，再规划“尚未发生的未来”）：
        1) NEW / WAITING / WAITING_FOR_FLIGHT / WAITING_FOR_TRAIN 等：用 scenarios 做期望成本比较
        2) TRANSFERRING：用实际观测判断是否“航空外包”（错过航班）
        """

        # ---------- A. 已在换装中：用实际观测做二阶段调整 ----------
        if order.status == OrderStatus.TRANSFERRING:
            # 没有航班（本来就外包）-> 不在这里做重复决策
            if not order.assigned_flight:
                return None
            flight = flights.get(order.assigned_flight)

            # # 已观测：换装完成时间晚于航班实际起飞 -> 必然错过 -> 航空外包（部分联运）
            if order.transfer_complete_time > flight.actual_departure:
                cost = self._calc_cost_for_case(
                    case="RAIL_PLUS_AIR_OUTSOURCE",
                    order=order,
                    train_id=order.assigned_train,
                    flight_id=order.assigned_flight,
                    transfer_time=order.transfer_duration,
                    actual_pickup_time=order.actual_pickup_time,
                )
                return self._make_decision(
                    mode=TransportMode.INTERMODAL,
                    train=order.assigned_train,
                    flight=None,  # flight=None 表示航空段外包
                    cost=cost,
                )

            return None  # 能赶上就无需调整（继续原计划）

        # ---------- B. 新订单/等待类订单：做“期望成本”优化 ----------
        adhoc_cost = self._calc_cost_for_case(case="DIRECT_ADHOC", order=order)

        routes = self._find_feasible_routes(order, trains, flights, indexes, current_time)
        if not routes:
            return self._make_decision(TransportMode.AD_HOC, cost=adhoc_cost)

        best_route = None
        min_expected_cost = float("inf")

        for route in routes:
            exp_cost = self._expected_cost_over_scenarios(order, route, scenarios)
            if exp_cost < min_expected_cost:
                min_expected_cost = exp_cost
                best_route = route

        # 期望联运 < 直接外包 -> 选联运；否则全程外包
        if min_expected_cost < adhoc_cost and best_route is not None:
            return self._make_decision(
                TransportMode.INTERMODAL, best_route.train_id, best_route.flight_id, min_expected_cost
            )
        else:
            return self._make_decision(TransportMode.AD_HOC, cost=adhoc_cost)

    # =========================================================
    # 场景期望成本：内部使用统一的三情形成本函数
    # =========================================================
    def _expected_cost_over_scenarios(self, order: Order, route: RouteCandidate, scenarios: List[Scenario]) -> float:
        total_cost = 0.0

        sch_order_pickup = route.sch_order_pickup
        sch_arrival_at_hub = route.train_arrival_base

        for sc in scenarios:
            # 1) 生成该场景下的“实际轨迹”
            accumulated_delays = sc.train_delays.get(route.train_id, {})
            origin_delay = accumulated_delays.get(order.origin, 0.0)
            sim_pickup_time = sch_order_pickup + origin_delay

            hub_delay = list(accumulated_delays.values())[-1] if accumulated_delays else 0.0
            sim_transfer_done = sch_arrival_at_hub + hub_delay + sc.transfer_time

            f_delay = sc.flight_delays.get(route.flight_id, 0.0)
            sim_flight_dep = route.flight_departure_base + f_delay
            sim_flight_arr = sim_flight_dep + route.flight_travel_time

            # 2) 用统一成本函数覆盖两种联运结果：
            if sim_transfer_done <= sim_flight_dep:
                cost = self._calc_cost_for_case(
                    case="FINAL_INTERMODAL",
                    order=order,
                    train_id=route.train_id,
                    flight_id=route.flight_id,
                    transfer_time=sc.transfer_time,
                    actual_pickup_time=sim_pickup_time,
                    transfer_complete_time=sim_transfer_done,
                    actual_flight_time=sim_flight_dep,
                    actual_delivery_time=sim_flight_arr,
                )
            else:
                cost = self._calc_cost_for_case(
                    case="RAIL_PLUS_AIR_OUTSOURCE",
                    order=order,
                    train_id=route.train_id,
                    flight_id=route.flight_id,
                    transfer_time=sc.transfer_time,
                    actual_pickup_time=sim_pickup_time,
                )

            total_cost += cost

        return total_cost / max(len(scenarios), 1)

    def _calc_cost_for_case(
        self,
        case: str,
        order: Order,
        train_id: Optional[str] = None,
        flight_id: Optional[str] = None,
        transfer_time: Optional[float] = None,
        actual_pickup_time: Optional[float] = None,
        transfer_complete_time: Optional[float] = None,
        actual_flight_time: Optional[float] = None,
        actual_delivery_time: Optional[float] = None,
    ) -> float:
        """
        case:
          - "DIRECT_ADHOC"              : 直接外包
          - "FINAL_INTERMODAL"          : 空铁联运（成功衔接航班）
          - "RAIL_PLUS_AIR_OUTSOURCE"   : 铁路运输 + 航空外包（错过航班/不衔接）
        """

        if case == "DIRECT_ADHOC":
            return self.cost_calc.calculate_direct_adhoc_cost(order)

        if case == "FINAL_INTERMODAL":
            return self.cost_calc.calculate_final_intermodal_cost(
                order,
                assigned_train=train_id,
                assigned_flight=flight_id,
                transfer_duration=transfer_time,
                actual_pickup_time=actual_pickup_time,
                transfer_complete_time=transfer_complete_time,
                actual_flight_time=actual_flight_time,
                actual_delivery_time=actual_delivery_time,
            )

        if case == "RAIL_PLUS_AIR_OUTSOURCE":
            # 注意：这里仍然走 partial_intermodal_cost；你在外部用 flight=None 来表达“航空外包”
            return self.cost_calc.calculate_partial_intermodal_cost(
                order,
                assigned_train=train_id,
                assigned_flight=flight_id,
                transfer_duration=transfer_time,
                actual_pickup_time=actual_pickup_time,
            )

        raise ValueError(f"Unknown cost case: {case}")

    # =========================================================
    # 其余函数
    # =========================================================
    def _find_feasible_routes(self, order: Order, trains: Dict, flights: Dict,
                             indexes: Dict, current_time: float) -> List[RouteCandidate]:
        candidates = []

        possible_trains = indexes["train_origin"].get(order.origin, set())
        possible_flights = indexes["flight_dest"].get(order.destination, set())

        for tid in possible_trains:
            train = trains.get(tid)
            # --- 检查 1: 列车容量 ---
            if not train or train.remaining_capacity < order.volume: continue
            # --- 检查 2: 列车时间 ---
            if train.sch_departure < order.earliest_pickup: continue

            station = train.station_map.get(order.origin)
            sch_order_pickup = train.sch_departure + station.travel_time_acc

            for fid in possible_flights:
                flight = flights.get(fid)
                # --- 检查 3: 航班容量 ---
                if not flight or flight.remaining_capacity < order.volume: continue
                # --- 检查 4: 航班时间 ---
                transfer_time_mean = 75
                if flight.sch_departure < train.sch_arrival_at_hub + transfer_time_mean: continue

                # 如果通过所有检查，加入候选
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

    def _make_decision(self, mode: TransportMode, train=None, flight=None, cost=0.0):
        return {"mode": mode.name, "train": train, "flight": flight, "cost": cost}

    def _update_stats(self, stats, decision):
        if decision["mode"] == "INTERMODAL":
            stats["intermodal"] += 1
        else:
            stats["adhoc"] += 1