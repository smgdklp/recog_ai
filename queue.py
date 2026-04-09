import threading
import queue

class C_Reco:
    def __init__(self, custom=2):
        # 容器：pre_q,流式不阻塞队列，接受capture最多缓存5个
        self.pre_q = queue.Queue(maxsize=5)
        # cur_q,分发给下游的缓存容器
        self.cur_q = None
        # custom=2,分发数量
        self.custom = custom
        # count=0,消费者等候计数
        self.count = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        
    def pull(self, frame):
        # 将pre_q.put（frame）
        try:
            self.pre_q.put(frame, block=False)
        except queue.Full:
            pass
            
    def distri(self):
        # cur_q = pre_q.get
        self.cur_q = self.pre_q.get()
        
    def get(self):
        with self.lock:
            self.count += 1
            if self.count < self.custom:
                self.cond.wait()
            else:
                self.distri()
                self.count = 0
                self.cond.notify_all()
        return self.cur_q