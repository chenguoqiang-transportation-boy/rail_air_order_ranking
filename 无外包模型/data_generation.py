"""
数据生成模块：定义核心数据结构、延误模型及数据加载
"""
import bisect
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from scipy.stats import truncnorm, expon, triang
import os

# ============================================================================
# 枚举与常量
# ============================================================================

class TransportMode(Enum):
    INTERMODAL = 1  # 联运
    AD_HOC = 2      # 应急直达

class OrderStatus(Enum):
    NEW = 0
    WAITING = 1
    ON_TRAIN = 2
    TRANSFERRING = 3
    WAITING_FOR_FLIGHT = 4
    ON_FLIGHT = 5
    DELIVERED = 6
    DELAYED = 7
    AD_HOC_MODE = 8

# ============================================================================
# 核心实体 (Dataclasses)
# ============================================================================

@dataclass
class Costs:
    """系统成本参数配置"""
    transfer_per_min: float = 0.5   # 换装成本 (元/分)
    holding_per_hour: float = 0.5   # 仓储/等待成本 (元/kg·分钟)
    ad_hoc_train_unit: float = 800  # 列车外包 (元/kg)
    ad_hoc_flight_unit: float = 900 # 航班外包 (元/kg)
    delay_penalty_rate: float = 0.5 # 延误罚金 (元/分·kg)
    # delay_penalty_rate: float = 10 # 延误罚金 (元/分·kg)

@dataclass
class Order:
    """订单实体"""
    id: str
    origin: str
    destination: str
    volume: float
    earliest_pickup: float
    deadline: float
    status: OrderStatus = OrderStatus.NEW

    # 成本参数
    lambda_f: float = 0.0  # 计算total_direct_cost的一个参数
    total_direct_cost: float = 0.0

    # 决策结果
    assigned_train: Optional[str] = None
    assigned_flight: Optional[str] = None
    mode: Optional[TransportMode] = None

    # 时间记录 (Actual)
    actual_pickup_time: Optional[float] = None
    actual_transfer_start: Optional[float] = None
    transfer_duration: float = 75.0 # 默认值，会被覆盖
    transfer_complete_time: Optional[float] = None
    actual_flight_time: Optional[float] = None
    actual_delivery_time: Optional[float] = None

    # 结果
    total_cost: float = 0.0

@dataclass
class TrainStation:
    """列车站点信息"""
    train_id: str
    origin_seq_id: str
    origin_id: str
    origin_name: str
    travel_time: float
    trans_cost_to_d: float
    station_delay: float = 0.0
    travel_time_acc: float = 0.0
    actual_arrival: float = 0.0

@dataclass
class BaseResource:
    """资源基类"""
    id: str
    name: str
    capacity: float
    sch_departure: float
    remaining_capacity: float = field(init=False)
    actual_departure: Optional[float] = None
    actual_arrival: Optional[float] = None

    def __post_init__(self):
        self.remaining_capacity = self.capacity

@dataclass
class Train(BaseResource):
    stations: List[TrainStation] = field(default_factory=list)
    sch_arrival_at_hub: float = 0.0  # 计划到达枢纽时间

    def __post_init__(self):
        super().__post_init__() # 务必调用父类(BaseResource)的逻辑
        self.station_map = {s.origin_id: s for s in self.stations}  # 起点：TrainStation。通过起点id可以得到对应站点
        self.sch_arrival_at_hub = self.sch_departure + sum(s.travel_time for s in self.stations)

    def get_arrival_at(self, origin_id):
        """封装获取逻辑：给定站点ID，返回该列车在该站的实际到达/出发时间"""
        station = self.station_map.get(origin_id)
        if station:
            return station.actual_arrival
        raise ValueError(f"Train {self.id} does not stop at {origin_id}")

@dataclass
class Flight(BaseResource):
    destination_id: str = ""
    destination_name: str = ""
    travel_time: float = 0.0
    trans_cost: float = 0.0
    average_delay: float = 0.0
    worst_delay: float = 0.0

@dataclass
class Scenario:
    id: str
    probability: float
    train_delays: Dict[str, Dict[Any, float]]
    flight_delays: Dict[str, float]
    transfer_time: float

# ============================================================================
# 逻辑类
# ============================================================================

class DelayModel:
    """延误生成模型"""
    # 【新增】敏感性系数，默认为 1.0 (正常延误)。
    # 设置为 0.0 即无延误，设置为 2.0 即双倍延误。
    SENSITIVITY_FACTOR = 1.0

    @staticmethod
    def sample_flight_delay(average_delay: float = 25, worst_delay: float = 60) -> float:
        # 【修改】将输入参数乘以系数
        avg = average_delay * DelayModel.SENSITIVITY_FACTOR
        worst = worst_delay * DelayModel.SENSITIVITY_FACTOR

        if worst == 0: return 0.0  # 防止除零错误

        c = max(0, min(1, avg / worst))
        # loc=0, scale=worst 决定了延误的范围
        return triang.rvs(c=c, loc=0, scale=worst)

    @staticmethod
    def sample_train_delay(mean: float = 15) -> float:
        # 【修改】均值乘以系数
        adjusted_mean = mean * DelayModel.SENSITIVITY_FACTOR
        if adjusted_mean == 0: return 0.0
        return expon.rvs(scale=adjusted_mean)

    @staticmethod
    def sample_transfer_time(mean: float = 75, std: float = 20, lower_bound: float = 60) -> float:
        # 【修改】换乘时间也可以受延误系数影响，或者保持不变，视你的研究需求而定。
        # 这里假设换乘时间的波动也变大
        adj_mean = mean * DelayModel.SENSITIVITY_FACTOR
        adj_std = std * DelayModel.SENSITIVITY_FACTOR
        adj_lower = lower_bound * DelayModel.SENSITIVITY_FACTOR

        if adj_std == 0: return adj_mean  # 防止错误

        a = (adj_lower - adj_mean) / adj_std
        return truncnorm.rvs(a, np.inf, loc=adj_mean, scale=adj_std)


class DataLoader:
    """通用数据加载器"""
    def __init__(self, data_dir: str = './data/Instance a/ '):
        self.data_dir = data_dir

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path, delimiter=';')

        # 自动将所有ID列转换为字符串
        id_columns = [col for col in df.columns if 'id' in col.lower() or 'Id' in col]
        for col in id_columns:
            df[col] = df[col].astype(str)
        return df

    def load_all(self, train_file: str, station_file: str, flight_file: str, cargo_file: str) -> Dict[str, pd.DataFrame]:
        return {
            'origin': self._load_csv('origin.csv'),
            'destination': self._load_csv('destination.csv'),
            'train': self._load_csv(train_file),
            'train_station': self._load_csv(station_file),
            'flight': self._load_csv(flight_file),
            'cargo': self._load_csv(cargo_file)
        }

class OrderGenerator:
    def __init__(self, cargo_df: pd.DataFrame):
        self.cargo_df = cargo_df
        self.counter = 0

    def generate(self) -> List[Order]:
        orders = []
        for _, row in self.cargo_df.iterrows():
            self.counter += 1
            orders.append(Order(
                id=f"O{self.counter}",
                origin=row['originId'],
                destination=row['destinationId'],
                volume=row['volume'],
                earliest_pickup=row['e_f'],
                deadline=row['l_f'],
                lambda_f=row['lambda_f'],
                total_direct_cost=row['total_direct_cost']
            ))
        return orders

class NetworkInitializer:
    @staticmethod
    def init_trains(train_df: pd.DataFrame, station_df: pd.DataFrame, origin_df: pd.DataFrame) -> Dict[str, Train]:
        trains = {}
        origin_map = dict(zip(origin_df['id'], origin_df['name']))

        # 预先分组以减少循环
        grouped_stations = station_df.groupby('trainId')

        for _, row in train_df.iterrows():
            t_id = row['id']
            stations = []
            if row['id'] in grouped_stations.groups:
                st_data = grouped_stations.get_group(row['id'])
                for _, st_row in st_data.iterrows():
                    stations.append(TrainStation(
                        train_id=t_id,
                        origin_seq_id=st_row['originSeqId'],
                        origin_id=st_row['originId'],
                        origin_name=origin_map.get(st_row['originId'], f"S_{st_row['originId']}"),
                        travel_time=st_row['travel_time'],
                        trans_cost_to_d=st_row['trans_cost_to_d']
                    ))

            trains[t_id] = Train(
                # id=t_id, name=row['name'], capacity=row['capacity']*0.1,
                id=t_id, name=row['name'], capacity=row['capacity'],
                sch_departure=row['sch_departure'], stations=stations
            )
        return trains

    @staticmethod
    def init_flights(flight_df: pd.DataFrame, dest_df: pd.DataFrame) -> Dict[str, Flight]:
        flights = {}
        dest_map = dict(zip(dest_df['id'], dest_df['name']))
        for _, row in flight_df.iterrows():
            f_id = row['id']
            flights[f_id] = Flight(
                # id=f_id, name=row['name'], capacity=row['capacity']*0.1,
                id=f_id, name=row['name'], capacity=row['capacity'],
                sch_departure=row['sch_departure'], destination_id=row['destinationId'],
                destination_name=dest_map.get(row['destinationId'], f"D_{row['destinationId']}"),
                travel_time=row['travel_time'], trans_cost=row['trans_cost'],
                average_delay=row['average_delay'], worst_delay=row['worst_delay']
            )
        return flights

class ScenarioGenerator:
    def __init__(self, delay_model: DelayModel):
        self.delay_model = delay_model

    def generate(self, trains: Dict[str, Train], flights: Dict[str, Flight],
                 num_scenarios: int, current_time: float,
                 train_actuals: Dict) -> List[Scenario]:
        """
        生成场景：对于已发生的部分使用实际值，未发生的部分采样
        """
        scenarios = []
        prob = 1.0 / max(1, num_scenarios)

        for s in range(num_scenarios):
            # 1. 列车延误 (混合: 历史实际 + 未来采样)
            train_delays = {}
            for t_id, train in trains.items():
                stations = train.stations
                n_stations = len(stations)

                # A. 获取历史延误 (List)
                # train_actuals 结构: {train_id: ([station_delay], [actual_arrival])}
                history_delays = []
                if t_id in train_actuals:
                    station_delays, actual_arrivals = train_actuals[t_id]
                    idx = bisect.bisect_left(actual_arrivals, current_time)  # 找到当前时刻之前的最后一个站点索引
                    history_delays = station_delays[:idx]

                # B. 补齐未来延误 (List)
                missing_count = n_stations - len(history_delays)
                future_delays = [self.delay_model.sample_train_delay() for _ in range(missing_count)]

                # C. 合并得到完整线路的单站延误列表
                all_delays = history_delays + future_delays

                # D. 转换为 {StationID: 累积延误} 方便后续直接查
                # 这一步是必须的，因为 [5, 2] 表示第1站延误5分，第2站增加2分
                # 那么第2站的实际发车时间是延误了 5+2=7 分钟
                accumulated_delays = {}
                current_acc = 0.0
                for i, d in enumerate(all_delays):
                    current_acc += d
                    # 找到对应的 Station ID
                    st_id = stations[i].origin_id
                    accumulated_delays[st_id] = current_acc  # 每个站点的累计延误
                train_delays[t_id] = accumulated_delays  # 每个列车的累计延误

            # 2. 换装时间【可以考虑是否使用真实的换装时间】
            s_transfer = self.delay_model.sample_transfer_time()

            # 3. 航班延误【因为暂时不需要航班延误的参数来决定是否改外包，所以可以不用添加真实数据。】
            flight_delays = {
                f_id: self.delay_model.sample_flight_delay(f.average_delay, f.worst_delay)
                for f_id, f in flights.items()
            }

            scenarios.append(Scenario(s, prob, train_delays, flight_delays, s_transfer))

        return scenarios
