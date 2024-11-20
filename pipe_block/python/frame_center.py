import numpy as np
import cv2
from typing import Any, List, Tuple, Deque
from collections import deque
import itertools
import time
from threading import Thread
from loguru import logger

class FrameCenter:
    def __init__(self, video_path:str, max_batch:int=1, start_second:int=None, frame_step:int=1, fps:int=None, async_mode:bool=False, io_backend:str="FFMPEG")->None:
        self.backend = cv2.CAP_FFMPEG if io_backend=="FFMPEG" else cv2.CAP_GSTREAMER
        self.read_gap = frame_step - 1
        self.max_batch = max_batch
        self.frame_bfr = deque()
        self.current_pipe = None
        self.base_frame_ID = 1
        self.video_path=video_path
        self.setting_fps = fps
        self.start_second = start_second
        self.on = False
        self.async_mode = async_mode
        self.thread_worker = None
        if self.async_mode:
            self.on = True
            self.thread_worker = Thread(target=self.Async_Load)
            self.thread_worker.start()
    
    def init_capture(self)->None:
        self.cap = cv2.VideoCapture(self.video_path, self.backend)
        if not self.cap.isOpened():
           return
        self.Metadata = {
            'fps':int(self.cap.get(cv2.CAP_PROP_FPS)) if self.setting_fps is None else self.setting_fps,
            'height':self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'width':self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        }
        if self.start_second is not None:
            for _ in range(int(self.start_second*self.Metadata['fps'])):
                self.cap.read()

    # 從影像緩存 (frame_bfr) 中分配一批影像幀數據，負責管理幀的批次大小，並處理幀數據的 ID 分配
    def Allocate(self)->Tuple[Deque[np.ndarray], int, Deque[int], bool]:
        if len(self.frame_bfr) == 0:
            return deque(), 0, deque(), True
        if len(self.frame_bfr[0]) > self.max_batch:
            bfr = deque(itertools.islice(self.frame_bfr[0], self.max_batch))
            for _ in range(self.max_batch):
                self.frame_bfr[0].popleft()
            fids = deque(range(self.base_frame_ID, self.base_frame_ID+self.max_batch))
            self.base_frame_ID = self.base_frame_ID + self.max_batch
            return bfr, self.max_batch, fids, False
        
        bfr = self.frame_bfr[0].copy()
        self.frame_bfr[0].clear()
        fids = deque(range(self.base_frame_ID, self.base_frame_ID+len(bfr)))
        self.base_frame_ID = self.base_frame_ID + len(bfr)

        if len(self.frame_bfr) > 1 or self.current_pipe is None:
            self.frame_bfr.popleft()
            self.base_frame_ID = 1
            return bfr, len(bfr), fids, True
        return bfr, len(bfr), fids, False

    def Load(self)->None:
        if self.async_mode:
            logger.warning('Since async_mode is enabled. Synchronous "Load" method won\'t work.')
            return
        for _ in range(self.max_batch):
            if self.current_pipe is None:
                self.init_capture()
            ret, frame = self.cap.read()
            if not ret:
                self.current_pipe = None
                break
            if self.current_pipe is None:
                self.current_pipe = deque()
                self.frame_bfr.append(self.current_pipe)
            self.current_pipe.append(frame)
            for _ in range(self.read_gap):
                ret, _ = self.cap.read() if ret else (False, None)
            if not ret:
                self.current_pipe = None
                break

    def Async_Load(self)->None:
        self.init_capture()
        while self.on:
            ret, frame = self.cap.read()
            if not ret:
                self.current_pipe = None
                time.sleep(1)
                self.init_capture()
            else:
                if self.current_pipe is None:
                    self.current_pipe = deque()
                    self.frame_bfr.append(self.current_pipe)
                self.current_pipe.append(frame)
                for _ in range(self.read_gap):
                    ret, _ = self.cap.read() if ret else (False, None)
    
    def Get(self, meta_type:int)->Any:
        return self.cap.get(meta_type)

    def Exit(self)->None:
        self.on = False
        if self.async_mode:
            logger.debug(f'====================thread shutdown====================')
            if self.thread_worker is not None and self.thread_worker.is_alive():
                self.thread_worker.join()
            logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
            logger.debug(f'=======================================================')
        self.cap.release()
