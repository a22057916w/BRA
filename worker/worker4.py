import os.path as osp
import sys
import cv2
import numpy as np
from loguru import logger
import pandas as pd
import math
import copy
import yaml

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
sys.path.append(osp.join(__proot__,  "third_party", "YOLOX"))
sys.path.append(osp.join(__proot__, "third_party", "ByteTrack", "yolox"))
sys.path.append(osp.join(__proot__, "exps"))

import compute_block as cmb
from tracker.byte_tracker import BYTETracker
from pipe_block import (
    FrameCenter,
    IPMer,
    EventFilter,
    Detector,
    AppearanceExtractor,
    BYTETracker_reid,
)

from .BaseWorker import BaseWorker

class Worker(BaseWorker):
    def __init__(
        self,
        vin_path, 
        view_cfg,
        worker_cfg,
        actual_framerate=None,
        reid=False,
        start_second=None,
        batch_size=1,
        enable_track=None,
        io_backend="FFMPEG"
    ):
        super().__init__(EOI=['LOSS_CAPTURE', 'GET_CAPTURE'])
        configStream = open(worker_cfg, 'r')
        self.config = yaml.safe_load(configStream)
        configStream.close()
        self.vcfg_path = view_cfg

        if 'enable' not in self.config['track_opt'] and enable_track is None:
            self.config['track_opt']['enable'] = True
        elif enable_track is not None:
            self.config['track_opt']['enable'] = enable_track

        self.fc_async_mode = self.config['general']['async_frame_center'] if 'async_frame_center' in self.config['general'] else False
        self.FCenter = FrameCenter(vin_path, max_batch=batch_size, start_second=start_second, fps=actual_framerate, async_mode=self.fc_async_mode, io_backend=io_backend)
        self.on = False
        self.fps = None
        self.reid = reid
        
        self.MDetector = None
        if self.config['mask']['pipe_enable']:
            self.MDetector = Detector(
                device=self.config['mask']['device'],
                exp_path=self.config['mask']['exp'],
                checkpoint=self.config['mask']['checkpoint'],
                fuse=self.config['mask']['fuse'],
                fp16=self.config['mask']['fp16'],
                cls_name=["without_mask", "with_mask", "mask_wear_incorrect"],
                legacy=self.config['mask']['legacy'],
                trt_mode=self.config['mask']['trt_mode'],
                trt_path=self.config['mask']['trt_path'],
                trt_batch=None if 'trt_batch' not in self.config['mask'] else self.config['mask']['trt_batch'],
                trt_workspace=None if 'trt_workspace' not in self.config['mask'] else self.config['mask']['trt_workspace'],
            )
        self.PDetector = None
        if self.config['person']['pipe_enable']:
            self.PDetector = Detector(
                device=self.config['person']['device'],
                exp_path=self.config['person']['exp'],
                checkpoint=self.config['person']['checkpoint'],
                fuse=self.config['person']['fuse'],
                fp16=self.config['person']['fp16'],
                cls_name=['person'],
                legacy=self.config['person']['legacy'],
                trt_mode=self.config['person']['trt_mode'],
                trt_path=self.config['person']['trt_path'],
                trt_batch=None if 'trt_batch' not in self.config['person'] else self.config['person']['trt_batch'],
                trt_workspace=None if 'trt_workspace' not in self.config['person'] else self.config['person']['trt_workspace'],
            )
        self.track_parameter = {
            "track_thresh":self.config['track_opt']['track_thresh'],
            "track_buffer":self.config['track_opt']['track_buffer'],
            "match_thresh":self.config['track_opt']['match_thresh'],
            "aspect_ratio_thresh":self.config['track_opt']['aspect_ratio_thresh'],
            "min_box_area":self.config['track_opt']['min_box_area'],
            "mot20":False,
        }
        self.byteTracker = None
        if self.reid:
            self.appearanceExtractor = AppearanceExtractor(
                device=self.config['person']['reid_opt']['device'],
                cfg_path=self.config['person']['reid_opt']['cfg'],
                checkpoint=self.config['person']['reid_opt']['checkpoint']
            )
        self.ipmer = IPMer(self.vcfg_path)
        self.washFilter = None
        self.person_count_metrics = None
        self.mask_metrics = None
        self.distance_metrics = None
        self.wash_correct_metrics = None


    def lazy_init(self):
        if self.config['track_opt']['enable'] and self.reid:
            self.byteTracker = BYTETracker_reid(type('',(object,),self.track_parameter)(), frame_rate=self.fps)
        else:
            self.byteTracker = BYTETracker(type('',(object,),self.track_parameter)(), frame_rate=self.fps)
        self.washFilter = EventFilter(self.vcfg_path, self.config['general']['record_life'], self.fps)
        self.person_count_metrics = cmb.ACBlock(
            self.fps*self.config['general']['metrics_duration'],
            self.fps*self.config['general']['metrics_update_time'],
        )
        self.mask_metrics = cmb.ACBlock(
            self.fps*self.config['general']['metrics_duration'],
            self.fps*self.config['general']['metrics_update_time'],
        )
        self.distance_metrics = cmb.ACBlock(
            self.fps*self.config['general']['metrics_duration'],
            self.fps*self.config['general']['metrics_update_time'],
        )
        self.wash_correct_metrics = cmb.ACBlock(
            self.fps*self.config['general']['metrics_duration'],
            self.fps*self.config['general']['metrics_update_time'],
        )

    def _examWork(self):
        return True

    def _workFlow(self, dataToWork):
        # [LEVEL_0_BLOCK]
        FList, length, FIDs = dataToWork
        null_list = [None for _ in range(length)]
        packet = {
            'frames':FList,
            'fids':FIDs,
        }

        # [LEVEL_1_BLOCK] === INPUT -> frame
        person_outputs, person_infos = null_list, null_list
        if self.config['person']['pipe_enable']:
            person_outputs, person_infos = self.PDetector.detect(packet['frames'])
        packet['p_outputs'] = person_outputs
        packet['p_infos'] = person_infos

        mask_outputs, mask_infos = null_list, null_list
        if self.config['mask']['pipe_enable']:
            mask_outputs, mask_infos = self.MDetector.detect(packet['frames'])
        packet['m_outputs'] = mask_outputs
        packet['m_infos'] = mask_infos

        # [LEVEL_2_BLOCK] === INPUT -> LEVEL_1_PERSON_BLOCK result
        crowd_count = null_list
        if self.config['person']['pipe_enable']:
            if self.config['track_opt']['enable']:
                tracked_person = []
                for frame, p_output, p_info in zip(packet['frames'], packet['p_outputs'], packet['p_infos']):
                    if p_output is None:
                        tracked_person.append([])
                        continue
                    if self.reid:
                        feats = self.appearanceExtractor.extract_with_crop(frame, p_output, 0.1)
                        tracked_person.append(
                            copy.deepcopy(
                                self.byteTracker.update(
                                    p_output, 
                                    [p_info['height'], p_info['width']], 
                                    self.PDetector.test_size,
                                    feats
                                )
                            )
                        )
                        continue
                    tracked_person.append(
                        copy.deepcopy(
                            self.byteTracker.update(
                                p_output, 
                                [p_info['height'], p_info['width']], 
                                self.PDetector.test_size
                            )
                        )
                    )
                packet['tracked_person'] = tracked_person
                crowd_count = [len(t_person) for t_person in tracked_person]
            else:
                packet['tracked_person'] = [
                    self.PDetector.output_tidy(
                        p_output.cpu().numpy(), p_info, 
                        self.config['track_opt']['match_thresh']
                    )['person']
                for p_output, p_info in zip(person_outputs, person_infos)]
                crowd_count = [len(person) for person in packet['tracked_person']]
        packet['crowd_count'] = crowd_count

        mask_wearing_count, no_mask_count = null_list, null_list
        if self.config['mask']['pipe_enable']:
            mask_wearing_count, no_mask_count = [], []
            for frame, m_output, m_info in zip(packet['frames'], packet['m_outputs'], packet['m_infos']):
                with_mask, without_mask = [], []
                if m_output is not None:
                    mask_out = m_output
                    # mask_out = mask_outputs[0].cpu()
                    out = self.MDetector.output_tidy(mask_out, m_info, 0.5)
                    # self.MDetector.visual_rp(frame, mask_out, m_info, 0.5)
                    with_mask = out['with_mask']+out['mask_wear_incorrect']
                    without_mask = out['without_mask']
                mask_wearing_count.append(len(with_mask))
                no_mask_count.append(len(without_mask))
        packet['mask_wearing_count'] = mask_wearing_count
        packet['no_mask_count'] = no_mask_count
            
        # [LEVEL_3_BLOCK] === INPUT -> tracked ID and tracked person
        # (HAND)
        no_hand_washing_count, hand_washing_wrong_count, hand_washing_correct_count = null_list, null_list, null_list
        if self.config['person']['pipe_enable']:
            no_hand_washing_count, hand_washing_wrong_count, hand_washing_correct_count = [], [], []
            for frame, t_person in zip(packet['frames'], packet['tracked_person']):
                notWashIds, wrongWashIds, correctWashIds = [], [], []
                if self.config['track_opt']['enable']:
                    notWashIds, wrongWashIds, correctWashIds, c_table, b_table = self.washFilter.work(t_person)
                    self.washFilter.visualize(frame, t_person)
                no_hand_washing_count.append(len(notWashIds))
                hand_washing_wrong_count.append(len(wrongWashIds))
                hand_washing_correct_count.append(len(correctWashIds))
        packet['no_hand_washing_count'] = no_hand_washing_count
        packet['hand_washing_wrong_count'] = hand_washing_wrong_count
        packet['hand_washing_correct_count'] = hand_washing_correct_count

        # (DISTANCE)
        social_distance, distance_segment_count = null_list, null_list
        if self.config['person']['pipe_enable']:
            social_distance, distance_segment_count = [], []
            for frame, t_person in zip(packet['frames'], packet['tracked_person']):
                if self.config['track_opt']['enable']:
                    bottom_center_points = np.asarray(
                        [(p.tlwh[0]+p.tlwh[2]/2, p.tlwh[1]+p.tlwh[3]) for p in t_person]
                    )
                else:
                    bottom_center_points = np.asarray(
                        [((p['bbox'][0]+p['bbox'][2])/2, p['bbox'][3]) for p in t_person]
                    )
                distance = self.ipmer.calc_bev_distance(bottom_center_points)
                social_distance.append(distance.sum() / 2)
                distance_segment_count.append(len(bottom_center_points) * (len(bottom_center_points)-1) / 2)
                IPMer.draw_warning_line(frame, bottom_center_points, distance, color=(0, 0, 255), thickness=6, floor=0, ceil=150.0, equal=True)
                IPMer.draw_warning_line(frame, bottom_center_points, distance, color=(0, 133, 242), thickness=2, floor=150.0, ceil=250.0, equal=False)
                for point in bottom_center_points:
                    cv2.circle(
                        frame, 
                        (np.int32(point[0]), np.int32(point[1])), 
                        6, (0, 255, 0),
                        -1, 8, 0
                    )
        packet['social_distance'] = social_distance
        packet['distance_segment_count'] = distance_segment_count

        # [LEVEL_4_BLOCK] === INPUT -> assessment result
        raw_data, short_data = [], []
        for (fid, frame,
            crowdCount,
            sdist, distSegN,
            maskCount, noMaskCount,
            noWashCount, wrongWashCount, correctWashCount
            ) in zip(
            packet['fids'], packet['frames'],
            packet['crowd_count'],
            packet['social_distance'], packet['distance_segment_count'],
            packet['mask_wearing_count'], packet['no_mask_count'],
            packet['no_hand_washing_count'], packet['hand_washing_wrong_count'], packet['hand_washing_correct_count']
        ):
            avg_dist, p_dist = None, None
            mask_ratio, p_mask_ratio = None, None
            p_crowdCount, is_new = None, False
            p_wash = None
            if sdist is not None:
                avg_dist = sdist / distSegN if distSegN != 0 else 0
                p_dist, is_new = self.distance_metrics.step(sdist, distSegN)
            if maskCount is not None:
                mask_ratio = maskCount / (maskCount + noMaskCount) if (maskCount + noMaskCount) != 0 else 0.0
                p_mask_ratio, is_new = self.mask_metrics.step(maskCount, maskCount+noMaskCount)
            if crowdCount is not None:
                p_crowdCount, is_new = self.person_count_metrics.step(crowdCount, 1)
            if self.config['track_opt']['enable'] and correctWashCount is not None:
                p_wash, is_new = self.wash_correct_metrics.step(correctWashCount, noWashCount+wrongWashCount+correctWashCount)
            texts = [
                f'Real-time',
                f'+ Person Count: {f"{crowdCount:d}" if crowdCount is not None else "not support"}',
                f'+ Average Distance: {f"{avg_dist / 100:.2f} m" if avg_dist is not None else "not support"}',
                f'+ Mask Ratio: {f"{mask_ratio * 100:.2f}% ({maskCount})" if mask_ratio is not None else "not support"}',
            ]
            cmb.VISBlockText(
                frame,
                texts, (0, 0),
                ratio=1, thickness=3,
                fg_color=(0 ,0 ,0), bg_color=(255, 255, 255, 0.4),
                point_reverse=(False,True)
            )
            texts = [
                f'Period-time',
                # f'fid: {fid}',
                # f'self.person_count_metrics.counter: {self.person_count_metrics.counter}',
                # f'self.person_count_metrics.AC_END: {self.person_count_metrics.AC_END}',
                # f'self.mask_metrics.counter: {self.mask_metrics.counter}',
                # f'self.mask_metrics.AC_END: {self.mask_metrics.AC_END}',
                # f'self.overall_metrics["mask"]: {self.overall_metrics["mask"]}',
                # f'self.wash_correct_metrics.counter: {self.wash_correct_metrics.counter}',
                # f'self.wash_correct_metrics.AC_END: {self.wash_correct_metrics.AC_END}',
                # f'self.wash_correct_metrics.period_buffer[\'denominator\']: {self.wash_correct_metrics.period_buffer["denominator"]}',
                f'+ Person Count: {f"{p_crowdCount:.2f}" if p_crowdCount is not None else "not support"}',
                f'+ Social Distance: {f"{p_dist / 100:.2f} m" if p_dist is not None else "not support"}',
                f'+ Mask Ratio: {f"{p_mask_ratio * 100:.2f}%" if p_mask_ratio is not None else "not support"}',
                f'+ Correct Wash: {f"{p_wash * 100:.2f} %" if p_wash is not None else "not support"}',
            ]
            cmb.VISBlockText(
                frame,
                texts, (0, 0),
                ratio=1, thickness=3,
                fg_color=(255, 255, 255), bg_color=(0 ,0 ,0, 0.4),
                point_reverse=(True,True)
            )

            hours, remain_second_frame = math.floor(fid / (3600*self.fps)), fid % (3600*self.fps)
            minutes, remain_second_frame = math.floor(remain_second_frame / (60*self.fps)), remain_second_frame % (60*self.fps)
            seconds, remain_second_frame = math.floor(remain_second_frame / (1*self.fps)), remain_second_frame % (1*self.fps)
            ms = math.floor(remain_second_frame * 1000 / self.fps)
            raw_data.append({
                'frame_id': fid,
                'play_time': f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}",
                'crowd_count': crowdCount,
                'social_distance': sdist, 'distance_segment_count': distSegN,
                'mask_wearing_count': maskCount, 'no_mask_count': noMaskCount,
                'no_hand_washing_count': noWashCount, 'hand_washing_wrong_count': wrongWashCount, 'hand_washing_correct_count': correctWashCount
            })
            if is_new:
                short_data.append({
                    'end_frame_id':fid,
                    'end_play_time':f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}",
                    'person_count':p_crowdCount,
                    'social_distance':p_dist,
                    'mask_ratio':p_mask_ratio,
                    'correct_wash':p_wash,
                })
        packet['raw_data'] = raw_data
        packet['short_data'] = short_data

        # [LEVEL_5_BLOCK] === OUTPUT
        return packet['fids'], packet['frames'], packet['raw_data'], packet['short_data']

    def _conditionWork(self):
        pass

    def _preparatory(self):
        if not self.fc_async_mode:
            self.FCenter.Load()
        data = self.FCenter.Allocate()
        eof = data[-1]
        length = data[1]
        logger.debug(f'Allocate {length} elements from FCenter.')
        logger.debug(f'{len(self.FCenter.frame_bfr[0]) if len(self.FCenter.frame_bfr) > 0 else 0} frames left in the buffer of this capture.')
        dataToWork = data[:-1]
        if length > 0 and not self.on:
            self.on = True
            self._signal('GET_CAPTURE')
            self.fps = self.FCenter.Metadata['fps']
            self.lazy_init()
        if eof and self.on:
            self.on = False
            self._signal('LOSS_CAPTURE')
        return dataToWork, length > 0

    def _endingWork(self):
        self.FCenter.Exit()


    def get_video_time(self):
        return self.FCenter.Get(cv2.CAP_PROP_POS_MSEC) / 1000
