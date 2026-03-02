from data_generation import Order, Costs

class CostCalculator:
    """
    统一成本计算器
    """
    def __init__(self, costs: Costs, origin_df=None, state_provider=None):
        self.costs = costs
        # 建立 origin_id -> holding_cost 的快速查找表
        self.origin_holding_map = dict(zip(origin_df['id'], origin_df['holding_cost']))
        self.state = state_provider # 用于查找 Train/Flight 对象

    # ========================== 公共计算方法 (Public API) ==========================
    def calculate_unserved_until_horizon_cost(self, order: Order, horizon_time: float) -> float:
        """
        无联运线路/未送达：从 earliest_pickup 一直滞留到 horizon，
        且若超过 deadline，则延误罚金一直累计到 horizon。
        """
        if horizon_time is None:
            raise ValueError("horizon_time 不能为空")

        cost = 0.0

        # 1) 始发站滞留成本：earliest_pickup -> horizon
        start = order.earliest_pickup
        end = horizon_time
        cost += self._calculate_train_wait_cost(order.origin, start, end, order.volume)

        # 2) 延误罚金：deadline -> horizon（若 horizon > deadline）
        delay_mins = max(0.0, horizon_time - order.deadline)
        cost += self.costs.delay_penalty_rate * delay_mins * order.volume

        return cost


    def calculate_final_intermodal_cost(self, order,assigned_train,assigned_flight, transfer_duration, actual_pickup_time, transfer_complete_time, actual_flight_time, actual_delivery_time) -> float:
        """计算最终实际发生的联运成本"""
        cost = 0.0
        # 1. 运输成本
        cost += self._get_train_transport_cost(assigned_train, order.origin, order.volume)
        cost += self._get_flight_transport_cost(assigned_flight, order.volume)

        # 2. 换装成本
        cost += self.costs.transfer_per_min * transfer_duration

        # 3. 等待成本
        cost += self._calculate_train_wait_cost(order.origin, order.earliest_pickup, actual_pickup_time, order.volume) # 始发站
        cost += self._calculate_airport_wait_cost(transfer_complete_time, actual_flight_time, order.volume) # 机场

        # 4. 延误罚金
        delay_mins = max(0, actual_delivery_time - order.deadline)
        cost += self.costs.delay_penalty_rate * delay_mins * order.volume

        return cost

    def calculate_partial_intermodal_cost(self, order, assigned_train,assigned_flight, transfer_duration, actual_pickup_time) -> float:
        """计算部分联运成本（列车段完成，航班段外包）"""
        cost = 0.0
        # 列车段实发成本
        cost += self._get_train_transport_cost(assigned_train, order.origin, order.volume)
        cost += self.costs.transfer_per_min * transfer_duration
        cost += self._calculate_train_wait_cost(order.origin, order.earliest_pickup, actual_pickup_time, order.volume)

        # 航班段外包
        cost += self.calculate_flight_adhoc_cost(order.volume)
        return cost

    def calculate_direct_adhoc_cost(self, order: Order) -> float:
        return order.total_direct_cost

    def calculate_flight_adhoc_cost(self, volume: float) -> float:
        return self.costs.ad_hoc_flight_unit * volume

    # ========================== 内部辅助逻辑 ==========================

    def _get_train_transport_cost(self, train_id: str, origin_id: str, volume: float) -> float:
        train = self.state.trains.get(train_id)
        station = train.station_map.get(origin_id)
        return station.trans_cost_to_d * volume

    def _get_flight_transport_cost(self, flight_id: str, volume: float) -> float:
        flight = self.state.flights.get(flight_id)
        return flight.trans_cost * volume if flight else 0.0

    def _calculate_train_wait_cost(self, origin_id: str, start: float, end: float, volume: float) -> float:
        if start is None or end is None or end <= start: return 0.0
        duration_mins = end - start
        unit_cost = self.origin_holding_map.get(origin_id, self.costs.holding_per_hour)
        return unit_cost * volume * duration_mins

    def _calculate_airport_wait_cost(self, start: float, end: float, volume: float) -> float:
        if start is None or end is None or end <= start: return 0.0
        duration_mins = end - start
        return self.costs.holding_per_hour * volume * duration_mins
