import os
import os.path as osp
import sys
import cv2
import argparse
import numpy as np
import subprocess
import time
import re
import math
import importlib
import csv
import datetime
from loguru import logger
from collections import deque
from threading import Thread
__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)

def regex_time_second(arg_value):
    pattern = '^(\d*):(\d*):(\d*)(\.\d+)?$'
    group = re.findall(pattern, arg_value)
    if re.match("^\d+$", arg_value) is not None:
        return int(arg_value)
    elif len(group) == 0:
        raise argparse.ArgumentTypeError("invalid value")
    group = group[0]
    second = 0
    for v in group[:-1]:
        second *= 60
        if len(v)!=0:
            second += int(v)
    if len(group[-1]) > 1:
        return second + float(group[-1])
    return second

def make_parser():
    parser = argparse.ArgumentParser("simple block work flow")
    parser.add_argument(
        "-vin", "--video_input",
        type=str, required=True,
        help="source to worker"
    )
    parser.add_argument(
        "-vc", "--view_config",
        type=str, required=True,
        help="YAML format configuration to camera"
    )
    parser.add_argument(
        "-wc", "--worker_config",
        type=str, required=True,
        help="YAML format configuration to worker"
    )
    parser.add_argument(
        "-ss", "--start_second",
        type=regex_time_second,
        default=None,
        help="start second from input video "
    )
    parser.add_argument(
        "-d", "--duration",
        type=regex_time_second, 
        default=None,
        help="stop after duration(s)"
    )
    parser.add_argument(
        "-fps", "--fps",
        type=float, default=None,
        help="disable auto fps calculation"
    )
    parser.add_argument(
        "-vout", "--video_output",
        type=str, default=None,
        help="store worker visualize result"
    )
    parser.add_argument(
        "-vlog", "--vout_log",
        type=str, default="video_storing.log",
        help="log when store video result with ffmpeg"
    )
    parser.add_argument(
        "-so", "--stream_output",
        type=str, default=None,
        help="media server location for pushing stream with ffmpeg"
    )
    parser.add_argument(
        "-slog", "--stream_log",
        type=str, default="stream_pushing.log",
        help="log when pushing stream with ffmpeg"
    )
    parser.add_argument(
        "-s", "--output_scale",
        type=str, default=None,
        help="scaling of output stream and stored file"
    )
    parser.add_argument(
        "-en", "--output_encoder",
        type=str, default="h264_nvenc",
        help="storing and pushing stream video encoder"
    )
    parser.add_argument(
        "-csv", "--write_to_csv",
        type=str, default=None,
        help="storing time sequence result as .csv"
    )
    parser.add_argument(
        '--reid', 
        action='store_true',
        help='using reid module to provide appearance features'
    )
    parser.add_argument(
        '-worker', '--worker_file',
        type=str, default="worker.worker3",
        help='specify which worker definition you want to use'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int, default=1,
        help='batch size for working with pipe'
    )
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='enable legacy mode to work on single input'
    )
    parser.add_argument(
        '--log_level',
        type=str, default='INFO',
        help='level for logger of work'
    )
    parser.add_argument(
        '--io_backend',
        type=str, default='FFMPEG',
        help='backend for processing io->FFMPEG or GSTREAMER'
    )
    parser.add_argument(
        '-ets', '--enable_task_scheduling',
        action='store_true',
        help="enable task scheduling (default FALSE)"
    )
    return parser

def make_ffmpeg_process(shape, fps, encoder, destination, log):
    command = ['ffmpeg',
        '-y', '-an',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{shape[0]}x{shape[1]}',
        '-framerate', f'{fps}',
        '-i', '-'
    ]
    target = destination.split('://') 
    if len(target)>1:
        stream_format = 'flv' if target[0]=='rtmp' else target[0]
        command.extend(['-f', f'{stream_format}'])
        if stream_format == 'flv':
            command.extend(['-flvflags','no_duration_filesize'])
    command.extend(
        [
            '-vcodec', encoder,
            destination,
            '-loglevel', 'verbose'
        ]
    )
    logger.debug(f'ffmpeg command is: {command}')
    # ref from https://stackoverflow.com/questions/5045771/python-how-to-prevent-subprocesses-from-receiving-ctrl-c-control-c-sigint
    if sys.platform.startswith('win'):
        return subprocess.Popen(
            command, shell=False,
            stdin=subprocess.PIPE,
            stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    import signal
    return subprocess.Popen(
        command, shell=False, 
        stdin=subprocess.PIPE,
        stdout=log, stderr=log,
        preexec_fn = lambda : signal.signal(signal.SIGINT, signal.SIG_IGN)
    )

# define the worker processing result stream pushing for thread
class publisher_ffmpeg:
    def __init__(self):
        self.opened = False
        self.pipe = deque()
        self.process = None
        self.null_frame = None
        self.thread_worker = Thread(target=self.run)
        self.sleep_duration = None

    def start(self, shape, fps, encoder, destination, log, null_frame):
        self.sleep_duration = 1/(fps*1.5)
        self.process = make_ffmpeg_process(shape, fps, encoder, destination, log)
        self.null_frame = null_frame.tobytes()
        self.opened = True
        self.thread_worker.start()
        logger.info(f'stream publisher to {destination} is starting.')

    def shutdown(self):
        logger.debug(f'====================thread shutdown====================')
        self.opened = False
        if self.thread_worker.is_alive():
            self.thread_worker.join()
        logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
        if self.process is not None:
            for _ in range(len(self.pipe)):
                bfr = self.pipe[0].tobytes()
                self.pipe.popleft()
                self.process.stdin.write(bfr)
            self.process.stdin.close()
            self.process.communicate()
            logger.debug(f'kill process of {self.process.args}')
            logger.debug(f'subprocess status? {self.process.poll()}')
            logger.debug(f'stdin of subprocess is close? {self.process.stdin.closed}')
        logger.debug(f'remain frames : {len(self.pipe)}')
        logger.debug(f'=======================================================')

    def run(self):
        bfr = None
        while self.opened:
            if len(self.pipe) > 0:
                bfr = self.pipe[0].tobytes()
                self.pipe.popleft()
                self.process.stdin.write(bfr)
            time.sleep(self.sleep_duration)

class publisher_gst:
    def __init__(self):
        logger.warning(f"Current version of BRA haven't supported gstreamer stream publisher. This function will work on monitor mode.")
        self.opened = False
        self.pipe = deque()
        self.writer = None
        self.gst_pattern = r"appsrc ! queue ! video/x-raw, format=BGR ! videoconvert ! autovideosink sync=false"
        self.null_frame = None
        self.thread_worker = Thread(target=self.run)
        self.sleep_duration = None

    def start(self, shape, fps, encoder, destination, log, null_frame):
        self.sleep_duration = 1/(fps*1.5)
        self.writer = cv2.VideoWriter(self.gst_pattern, cv2.CAP_GSTREAMER, 0, float(fps), shape)
        self.null_frame = null_frame
        self.opened = True
        self.thread_worker.start()
        logger.info(f'publisher to host monitor is starting.')

    def shutdown(self):
        logger.debug(f'====================thread shutdown====================')
        self.opened = False
        if self.thread_worker.is_alive():
            self.thread_worker.join()
        logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
        if self.writer is not None:
            for _ in range(len(self.pipe)):
                bfr = self.pipe[0]
                self.pipe.popleft()
                self.writer.write(bfr)
                cv2.waitKey(1)
            self.writer.release()
            logger.info(f'publisher to monitor is end.')
        logger.debug(f'remain frames : {len(self.pipe)}')
        logger.debug(f'=======================================================')

    def run(self):
        bfr = None
        while self.opened:
            if len(self.pipe) > 0:
                bfr = self.pipe[0]
                self.pipe.popleft()
                self.writer.write(bfr)
                cv2.waitKey(1)
            time.sleep(self.sleep_duration)

class segment_publisher_ffmpeg:
    def __init__(self):
        self.opened = True
        self.pipe = deque()
        self.process = None
        self.sleep_duration = 1
        self.thread_worker = Thread(target=self.run)
        self.thread_worker.start()
        self.ffmpeg_args = {
            'shape':None,
            'fps':None,
            'encoder':None,
            'destination':None,
            'log':None
        }
    
    def start(self, shape, fps, encoder, destination, log):
        self.sleep_duration = 1/(fps*1.5)
        self.ffmpeg_args = {
            'shape':shape,
            'fps':fps,
            'encoder':encoder,
            'destination':destination,
            'log':log
        }
        self.pipe.append(destination)
        logger.info(f'stream writer to {destination} is starting.')

    def shutdown(self):
        logger.debug(f'====================thread shutdown====================')
        self.opened = False
        if self.thread_worker.is_alive():
            self.thread_worker.join()
        logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
        if self.process is not None:
            for _ in range(len(self.pipe)):
                bfr = self.pipe[0]
                self.pipe.popleft()
                if bfr is None:
                    self.process.stdin.close()
                    self.process.communicate()
                    logger.info(f'stream writer to {self.ffmpeg_args["destination"]} is end.')
                elif isinstance(bfr, str):
                    self.ffmpeg_args['destination'] = bfr
                    self.process = make_ffmpeg_process(self.ffmpeg_args['shape'], self.ffmpeg_args['fps'], self.ffmpeg_args['encoder'], bfr, self.ffmpeg_args['log'])
                    logger.info(f'stream writer to {bfr} is starting.')
                elif isinstance(bfr, np.ndarray):
                    self.process.stdin.write(bfr.tobytes())
            if not self.process.stdin.closed:
                self.process.stdin.close()
            if self.process.poll() is None:
                self.process.communicate()
            logger.debug(f'kill process of {self.process.args}')
            logger.debug(f'subprocess status? {self.process.poll()}')
            logger.debug(f'stdin of subprocess is close? {self.process.stdin.closed}')
        logger.debug(f'remain frames : {len(self.pipe)}')
        logger.debug(f'=======================================================')

    def run(self):
        bfr = None
        while self.opened:
            if len(self.pipe) > 0:
                bfr = self.pipe[0]
                self.pipe.popleft()
                if bfr is None:
                    self.process.stdin.close()
                    self.process.communicate()
                    logger.info(f'stream writer to {self.ffmpeg_args["destination"]} is end.')
                elif isinstance(bfr, str):
                    self.ffmpeg_args['destination'] = bfr
                    self.process = make_ffmpeg_process(self.ffmpeg_args['shape'], self.ffmpeg_args['fps'], self.ffmpeg_args['encoder'], bfr, self.ffmpeg_args['log'])
                    logger.info(f'stream writer to {bfr} is starting.')
                elif isinstance(bfr, np.ndarray):
                    self.process.stdin.write(bfr.tobytes())
            time.sleep(self.sleep_duration)

class segment_publisher_gst:
    def __init__(self):
        self.opened = True
        self.pipe = deque()
        self.writer = None
        self.gst_pattern = r'appsrc ! video/x-raw, format=BGR ! queue ! videoconvert  %s filesink location=%s'
        self.cvt_enc_mux_bin = "!"
        self.sleep_duration = 1
        self.thread_worker = Thread(target=self.run)
        self.thread_worker.start()
        self.args = {
            'shape':None,
            'fps':None,
            'encoder':None,
            'destination':None,
            'log':None
        }
    
    def start(self, shape, fps, encoder, destination, log):
        if encoder[:2]=='nv':
            self.cvt_enc_mux_bin = f'! nvvidconv ! encodebin profile="matroskamux:{encoder}" !'
        else:
            self.cvt_enc_mux_bin = f'! encodebin profile="matroskamux:{encoder}" !'
        self.sleep_duration = 1/(fps*1.5)
        self.args = {
            'shape':shape,
            'fps':float(fps),
            'encoder':encoder,
            'destination':destination,
            'log':log
        }
        self.pipe.append(destination)
        logger.info(f'stream writer to {destination} is starting.')

    def shutdown(self):
        logger.debug(f'====================thread shutdown====================')
        self.opened = False
        if self.thread_worker.is_alive():
            self.thread_worker.join()
        logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
        if self.writer is not None:
            for _ in range(len(self.pipe)):
                bfr = self.pipe[0]
                self.pipe.popleft()
                if bfr is None:
                    self.writer.release()
                    logger.info(f'stream writer to {self.args["destination"]} is end.')
                elif isinstance(bfr, str):
                    self.args['destination'] = bfr
                    logger.info(f'Gstreamer Video Writer pattern is {self.gst_pattern % (self.cvt_enc_mux_bin, bfr)}.')
                    self.writer = cv2.VideoWriter(self.gst_pattern % (self.cvt_enc_mux_bin, bfr), cv2.CAP_GSTREAMER, 0, self.args['fps'], self.args['shape'])
                    if not self.writer.isOpened():
                        logger.error(f'writer to {bfr} is not reachable.')
                    logger.info(f'writer to {bfr} is starting.')
                elif isinstance(bfr, np.ndarray):
                    self.writer.write(bfr)
                    cv2.waitKey(1)
            if self.writer is not None and self.writer.isOpened():
                self.writer.release()
        logger.debug(f'remain frames : {len(self.pipe)}')
        logger.debug(f'=======================================================')
 
    def run(self):
        bfr = None
        while self.opened:
            if len(self.pipe) > 0:
                bfr = self.pipe[0]
                self.pipe.popleft()
                if bfr is None:
                    self.writer.release()
                    logger.info(f'stream writer to {self.args["destination"]} is end.')
                elif isinstance(bfr, str):
                    self.args['destination'] = bfr
                    logger.info(f'Gstreamer Video Writer pattern is {self.gst_pattern % (self.cvt_enc_mux_bin, bfr)}.')
                    self.writer = cv2.VideoWriter(self.gst_pattern % (self.cvt_enc_mux_bin, bfr), cv2.CAP_GSTREAMER, 0, self.args['fps'], self.args['shape'])
                    if not self.writer.isOpened():
                        logger.exception(f'writer to {bfr} is not reachable.')
                    logger.info(f'writer to {bfr} is starting.')
                elif isinstance(bfr, np.ndarray):
                    self.writer.write(bfr)
                    cv2.waitKey(1)
            time.sleep(self.sleep_duration)

class monitor_display:
    def __init__(self):
        self.opened = False
        self.pipe = deque()
        self.sleep_duration = 1
        self.thread_worker = Thread(target=self.run)
        self.sleep_duration = 0

    def start(self, shape, fps, encoder, destination, log, null_frame):
        self.sleep_duration = 1/(fps*1.5)
        self.null_frame = null_frame
        self.opened = True
        self.thread_worker.start()
        logger.info(f'[monitor_display] is starting.')

    def shutdown(self):
        logger.debug(f'===========Thread [monitor_display] shutdown===========')
        self.opened = False
        if self.thread_worker.is_alive():
            self.thread_worker.join()
        logger.debug(f'thread status? alive={self.thread_worker.is_alive()}')
        for _ in range(len(self.pipe)):
            self.pipe.popleft()
        cv2.destroyWindow("BRA display")
        logger.debug(f'remain frames : {len(self.pipe)}')
        logger.debug(f'=======================================================')

    def run(self):
        bfr = None
        while self.opened:
            if len(self.pipe) > 0:
                bfr = self.pipe[0]
                self.pipe.popleft()
                cv2.namedWindow("BRA display", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("BRA display", bfr)
                cv2.waitKey(1)
                time.sleep(self.sleep_duration)
            else:
                time.sleep(0.05)

@logger.catch
def main():
    args = make_parser().parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    # logger.add("work_dev.log", level="DEBUG", mode="w")
    if args.output_scale is not None:
        try:
            if re.match(
                "^\(([1-9]\d*|-1)\):\(([1-9]\d*|-1)\)$",
                args.output_scale
            ) is None:
                raise
            size = args.output_scale.split(':')
            if size[0] == size[1] == '(-1)':
                raise
            args.output_scale = (int(size[0][1:-1]), int(size[1][1:-1]))
        except:
            raise ValueError('The strings specified for scaling size are not correct. Please use "(width):(height)". And you can specify one of them to be -1 to automatically scale, but not both.')
    if args.legacy:
        logger.warning('Legacy is enable!')
    if args.io_backend != "FFMPEG" and args.io_backend != "GSTREAMER" and args.io_backend != "MIXFG" and args.io_backend != "MIXGF":
        logger.warning(f"io backend type:{args.io_backend} is not support. System will use default setting:FFMPEG.")
        args.io_backend = "FFMPEG"

    Worker = importlib.import_module(args.worker_file).Worker
    worker = Worker(
        vin_path=args.video_input,
        view_cfg=args.view_config,
        worker_cfg=args.worker_config,
        actual_framerate = args.fps,
        reid=args.reid,
        start_second = args.start_second,
        batch_size=args.batch_size,
        io_backend = 'FFMPEG' if args.io_backend=='FFMPEG' or args.io_backend=='MIXFG' else 'GSTREAMER',
        enable_task_scheduling = args.enable_task_scheduling
    )


    # default
    stored_size = None
    null_frame = None
    raw_data_record = []
    short_data_record = []
    time_point = None
    delta_time = None
    time_metrics = {'delta':0.0, 'frameN':0}

    vwriter_logFile = None
    video_writer = None
    if args.video_output is not None:
        vwriter_logFile = open(args.vout_log, 'w')
        if args.io_backend == "FFMPEG" or args.io_backend == "MIXGF" :
            video_writer = segment_publisher_ffmpeg()
        else:
            video_writer = segment_publisher_gst()
    
    stream_logFile = None
    stream_publisher = None
    if args.stream_output is not None:
        stream_logFile = open(args.stream_log, 'w')
        if args.stream_output[:7]=="MONITOR":
            stream_publisher = monitor_display()
        elif args.io_backend == "FFMPEG" or args.io_backend == "MIXGF" :
            stream_publisher = publisher_ffmpeg()
        else:
            stream_publisher = publisher_gst()
    
    if args.write_to_csv is not None:
        short_csv = None
        full_csv = None
    
    try:
        should_stop = False
        for result in worker:
            if should_stop and args.legacy:
                break
            events = worker.receive_signal()
            logger.trace(f'receive events is : {events}')
            if args.legacy and not worker.on and len(events)==0:
                raise Exception(f'Could not open file "{args.video_input}" for legacy mode')
            if 'GET_CAPTURE' in events:
                time_point = None
                should_stop = False
                logger.info('[GET_CAPTURE] signal was received.')
                frame_limit = None
                if args.duration is not None:
                    frame_limit = round(args.duration * worker.fps)
                    logger.info(f'Work will stop after {frame_limit} processed frame.')
                source_size = (
                    int(worker.FCenter.Metadata['width']),
                    int(worker.FCenter.Metadata['height'])
                )
                logger.debug(f'Capture dimension is : {source_size}')
                if stored_size is None:
                    if args.output_scale is None:
                        stored_size = source_size
                    else:
                        if args.output_scale[0] == -1:
                            ratio = args.output_scale[1] / source_size[1]
                            stored_size = (
                                int(source_size[0] * ratio),
                                args.output_scale[1]
                            )
                        elif args.output_scale[1] == -1:
                            ratio = args.output_scale[0] / source_size[0]
                            stored_size = (
                                args.output_scale[0],
                                int(source_size[1] * ratio)
                            )
                        else:
                            stored_size = args.output_size
                    null_frame = np.zeros((stored_size[1], stored_size[0], 3), np.uint8)
                    _text_size = cv2.getTextSize('WAIT SIGNAL...', cv2.FONT_HERSHEY_SIMPLEX, 6, 7)[0]
                    _pos = ((stored_size[0] - _text_size[0]) // 2 , (stored_size[1] + _text_size[1]) // 2)
                    cv2.putText(null_frame, 'WAIT SIGNAL...', _pos, cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 7)
                    logger.debug(f'Decide result dimension is {stored_size}.')
                shouldResize = source_size != stored_size

                datetime_tag = datetime.datetime.now().strftime(r'%Y%m%d_%H%M')
                if args.video_output is not None:
                    vwriter_filename = f'{args.video_output}{datetime_tag}.mkv' if not args.legacy else args.video_output
                    vwriter_filename = osp.abspath(vwriter_filename).replace(os.sep, '/')
                    video_writer.start(stored_size, worker.fps, args.output_encoder, vwriter_filename, vwriter_logFile)
                    logger.info(f'Decide the name of save video is : {vwriter_filename}')
                if args.write_to_csv is not None:
                    short_table_name = f'{args.write_to_csv}{datetime_tag}.csv' if not args.legacy else args.write_to_csv
                    short_csv = open(short_table_name, 'w', newline='')
                    logger.info(f'Decide the name of simple csv is : {short_table_name}')
                    full_table_name = osp.normpath(osp.join(osp.dirname(short_table_name), f'[raw]{osp.basename(short_table_name)}'))
                    full_csv = open(full_table_name, 'w', newline='')
                    logger.info(f'Decide the name of complete csv is : {full_table_name}')
                if args.stream_output is not None and not stream_publisher.opened:
                    stream_publisher.start(stored_size, worker.fps, args.output_encoder, args.stream_output, stream_logFile, null_frame)
            if result is not None:
                fids, frames, raw_data, short_data = result
                raw_data_record += raw_data
                short_data_record += short_data
                if len(fids) == 1:
                    logger.trace(f'Process progress is : {fids[0] / worker.fps:.3f}')
                else:
                    logger.trace(f'Process progress is : {fids[0] / worker.fps:.3f} to {fids[-1] / worker.fps:.3f}')
                for fid, frame in zip(fids, frames):
                    if frame_limit is not None and fid > frame_limit:
                        should_stop = True
                        logger.info(f'Reaching early stop => {frame_limit} frame.')
                        break
                    if shouldResize:
                        frame = cv2.resize(frame, stored_size)
                    if stream_publisher is not None:
                        stream_publisher.pipe.append(frame)
                    if video_writer is not None:
                        video_writer.pipe.append(frame)            
                        
                # analyst time consumption
                if time_point is not None:
                    delta_time = time.time() - time_point
                    time_point += delta_time
                    time_metrics['delta'] += delta_time
                    time_metrics['frameN'] += len(fids)
                else:
                    time_point = time.time()
                if time_metrics['frameN'] != 0:
                    logger.info(f'Process speed is : {time_metrics["frameN"]/time_metrics["delta"]:.4f} fps')
                if stream_publisher is not None:
                    logger.trace(f'stream publisher has {len(stream_publisher.pipe)} elements in pipe')
                if video_writer is not None:
                    logger.trace(f'video writer has {len(video_writer.pipe)} elements in pipe')
            elif len(events) == 0:
                if stream_publisher is not None and stream_publisher.opened and null_frame is not None:
                    for _ in range(int(worker.fps)):
                        stream_publisher.pipe.append(null_frame)
                time.sleep(1)
                logger.trace(f'sleep 1 second')
            if 'LOSS_CAPTURE' in events:
                logger.info('[LOSS_CAPTURE] signal was received.')
                if video_writer is not None:
                    video_writer.pipe.append(None)
                if args.write_to_csv is not None:
                    if len(short_data_record) < 1 or len(raw_data_record) < 1:
                        logger.warning('Worker may not support result recording.')
                    else:
                        writer = csv.DictWriter(short_csv, fieldnames=list(short_data_record[0].keys()))
                        writer.writeheader()
                        writer.writerows(short_data_record)
                        short_csv.close()
                        short_csv = None
                        writer = csv.DictWriter(full_csv, fieldnames=list(raw_data_record[0].keys()))
                        writer.writeheader()
                        writer.writerows(raw_data_record)
                        full_csv.close()
                        full_csv = None
                if args.legacy:
                    break
    except KeyboardInterrupt:
        worker._endingWork()
        logger.debug('Interrupt the work due to KeyboardInterrupt.')
    except Exception as e:
        logger.exception(e)
    finally:
        logger.info('Work finished.')
        if args.stream_output is not None:
            if stream_publisher is not None:
                stream_publisher.shutdown()
            stream_logFile.close()
        if args.video_output is not None:
            if video_writer is not None:
                video_writer.shutdown()
            vwriter_logFile.close()
        if args.write_to_csv:
            if short_csv is not None and not short_csv.closed and len(short_data_record) > 0:
                writer = csv.DictWriter(short_csv, fieldnames=list(short_data_record[0].keys()))
                writer.writeheader()
                writer.writerows(short_data_record)
                short_csv.close()
                short_csv = None
            if full_csv is not None and not full_csv.closed and len(raw_data_record) > 0:
                writer = csv.DictWriter(full_csv, fieldnames=list(raw_data_record[0].keys()))
                writer.writeheader()
                writer.writerows(raw_data_record)
                full_csv.close()
                full_csv = None

if __name__ == "__main__":
    main()