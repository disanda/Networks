import datetime
import enum
import logging#日志模块,参考https://www.cnblogs.com/Nicholas0707/p/9021672.html
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import colorlog#用于终端的日志有不同的颜色
import numpy as np
import torch
from tensorboardX import SummaryWriter

class MetricType(enum.IntEnum):
#枚举类型,是metric这个类的一个属性，代表metric的类型
#枚举键是左，值是右[1,2,3]
    Number = 1 #量化数字
    Loss = 2 #损失函数值
    Time = 3 #时间

class Metric(object):
    """Metric class for logger"""
    mtype_list: List[int] = list(map(int, MetricType))#就是枚举的值:[1,2,3]
    def __init__(self, mtype: MetricType, priority: int):
        if mtype not in self.mtype_list:
            raise Exception("mtype is invalid, %s".format(self.mtype_list))
        self.mtype: MetricType = mtype #是数字，损失函数值，还是时间
        self.params: Dict[str, Any] = {}#用于mtype为时间时，记录起始时间
        self.priority: int = priority
        self.value: Any = 0

def new_logging_module(name: str, log_file: Path) -> logging.Logger:
    # specify format
    log_format: str = "%(asctime)s - " "%(message)s"#log的格式
    bold_seq: str = "\033[1m"
    colorlog_format: str = f"{bold_seq} " "%(log_color)s " f"{log_format}"
    colorlog.basicConfig(format=colorlog_format)
    # init module
    logger: logging.Logger = logging.getLogger(name)#返回一个有名字的logging对象
    logger.setLevel(logging.DEBUG)#设置日志器将会处理的日志级别
    # add handler to output file setting
    fh = logging.FileHandler(str(log_file))#设置路径
    fh.setLevel(logging.DEBUG)#设置级别
    formatter = logging.Formatter(log_format)#设置格式
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

class Logger(object):
    """Logger for watchting some metrics involving training"""
#构造函数为两个路径,日志保存于这两个路径中
    def __init__(self, out_path: Path, tb_path: Path):
        # initialize logging module
        self._logger: logging.Logger = new_logging_module(__name__, out_path / "log")#添加logger对象，以及配对的handler对象
        self.path = out_path
        # logging metrics
        self.metrics: OrderedDict[str, Metric] = OrderedDict()#是一个元素值为Metric的字典:(name,metric)
        # tensorboard writer
        self.tf_writer: SummaryWriter = SummaryWriter(str(tb_path))
        # automatically add elapsed_time metric
        self.define("epoch", MetricType.Number, 5)
        self.define("iteration", MetricType.Number, 4)
        self.define("elapsed_time", MetricType.Time, -1)#级别最低显示在最后
    def define(self, name: str, mtype: MetricType, priority=0):
    #定义函数，输入名字和metric类型，以及这个metric的优先值
        metric: Metric = Metric(mtype, priority)
        if mtype == MetricType.Number:
            metric.value = 0
        elif mtype == MetricType.Loss:
            metric.value = []
        elif mtype == MetricType.Time:
            metric.value = 0
            metric.params["start_time"] = time.time()
        self.metrics[name] = metric
        self.metrics = OrderedDict(sorted(self.metrics.items(), key=lambda m: m[1].priority, reverse=True))
    def metric_keys(self) -> List[str]:
        return list(self.metrics.keys())
    def clear(self):
        for _, metric in self.metrics.items():
            if metric.mtype != MetricType.Loss:
                metric.value = []
    def update(self, name: str, value: Any):
    #这里的name相当于metric这个字典里面的键，值是metric对象
        m = self.metrics[name]
        if m.mtype == MetricType.Number:
            m.value = value
        elif m.mtype == MetricType.Loss:
            m.value.append(value)
        elif m.mtype == MetricType.Time:
            m.value = value - m.params["start_time"]
    def print_header(self):
    #这个是输出类，其中logger对象的提示类型有:debug，info，warning，error，critical
        log_string = ""
        for name in self.metrics.keys():
            log_string += "{:>13} ".format(name)#格式化输出name变量
        self._logger.info(log_string)
    def log(self):
        # display and save logs
        self.update("elapsed_time", time.time())
        log_strings: List[str] = []
        for k, m in self.metrics.items():
            if m.mtype == MetricType.Number:
                s = "{}".format(m.value)
            elif m.mtype == MetricType.Loss:
                if len(m.value) == 0:
                    raise Exception(f"Metric {k} has no values.")
                s = "{:0.3f}".format(sum(m.value) / len(m.value))
            elif m.mtype == MetricType.Time:
                _value = int(m.value)
                s = str(datetime.timedelta(seconds=_value))
            log_strings.append(s)
        log_string: str = ""
        for s in log_strings:
            log_string += "{:>13} ".format(s)
        self._logger.info(log_string)
    def log_tensorboard(self, x_axis_metric: str):
        # log MetricType.Loss metrics only
        if x_axis_metric not in self.metric_keys():
            raise Exception(f"No such metric: {x_axis_metric}")
        x_metric = self.metrics[x_axis_metric]
        if x_metric.mtype != MetricType.Number:
            raise Exception(f"Invalid metric type: {repr(x_metric.mtype)}")
        step = x_metric.value
        for name, metric in self.metrics.items():
            if metric.mtype != MetricType.Loss:
                continue
            if len(metric.value) == 0:
                raise Exception(f"Metric {name} has no values.")
            mean: float = sum(metric.value) / len(metric.value)
            self.tf_writer.add_scalar(name, mean, step)
    def tf_log_histgram(self, var, tag, step):
        var = var.clone().cpu().data.numpy()
        self.tf_writer.add_histogram(tag, var, step)
    def tf_log_image(self, x: torch.Tensor, step: int, tag: str):
        self.tf_writer.add_image(tag, x, step)
    def info(self, msg: str):
        self._logger.info(msg)
    def debug(self, msg: str):
        self._logger.debug(msg)

if __name__ == "__main__":
    import random
    from os.path import expanduser
    # init logger
    home = Path(expanduser("~"))
    out_path = home / "tmp/log"
    tfb_path = home / "tmp/log/tf"
    logger = Logger(out_path, tfb_path)
    print(logger.metric_keys())
    # add dummy metric
    logger.define("foo", MetricType.Number)
    logger.define("bar", MetricType.Loss)
    print(logger.metric_keys())
    logger.print_header()
    # update metric value and print log
    for i in range(10):
        logger.update("iteration", i % 2)
        logger.update("epoch", i // 2)
        logger.update("foo", random.randint(0, 100))
        logger.update("bar", random.randint(0, 10))
        logger.update("bar", random.randint(0, 10))
        logger.update("bar", random.randint(0, 10))
        time.sleep(1.0)
        logger.log()
