import pandas as pd
import yaml
from numpy import isnan
import cv2
import sys
import os.path as osp

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.append(__proot__)
from compute_block import PolygonExtension

class TriggerNode:
    def __init__(self, triggers):
        self.triggerList = triggers
    def __call__(self, inputs):
        chain = [True for _ in range(len(inputs))]
        for trigger in self.triggerList:
            chain = trigger(inputs, chain)
        return chain

class EventFilter:
    def __init__(self, configPath, record_life=5, fps=25):
        """
        初始化 EventFilter 類，讀取配置並設置相關的屬性。

        參數:
        - configPath (str): 配置文件的路徑，用於加載事件過濾的設定。
          配置文件中包含觸發條件的描述 (`trigger_description`) 和各種行為的閾值 (`threshold`)。
        - record_life (int, 默認值為 5): 指定對象在未檢測到目標行為後保留的幀數。
        - fps (int, 默認值為 25): 視頻的每秒幀數，用於將時間轉換為幀數。

        配置文件中的關鍵參數:
        - trigger_description (dict): 包含 `collector` 和 `base`，分別用於描述不同的觸發器節點配置。
        - threshold (dict): 包含三種狀態的時間閾值（以秒為單位）：
            - target: 目標行為所需的持續時間。
            - nonTarget: 非目標行為所需的持續時間，用於判斷對象不符合條件。
            - waiting: 等待狀態的持續時間，用於判斷對象是否繼續觀察。
        """
        configStream = open(configPath, 'r')
        self.config = yaml.safe_load(configStream)
        configStream.close()
        self.trigger_desc = self.config['trigger_description']
        # ===============================================================
        self.collector = []
        self.analyst_table = pd.DataFrame(columns=['targetCounter', 'nonTargetCounter'])
        self.base = []
        self.tracer_table = pd.DataFrame(columns=['type', 'deathCounter'])
        self.tracer_life = round(record_life * fps)
        # ===============================================================
        self.targetThreshold = round(self.trigger_desc['threshold']['target'] * fps)
        self.nonTargetThreshold = round(self.trigger_desc['threshold']['nonTarget'] * fps)
        self.waitingThreshold = round(self.trigger_desc['threshold']['waiting'] * fps)
        # ===============================================================
        self.deserialize(self.trigger_desc)

    def work(self, inputs):
        # inputs are list of object data with its properties

        '''
        All trigger functions define in collector and base will 
        filter input individually, and filtering result will be 
        represented as a boolean list.
        '''
        c_result = [node(inputs) for node in self.collector]
        b_result = [node(inputs) for node in self.base]

        '''
        Step 0: Start a new iteration
        '''
        self.analyst_table.iloc[:]['nonTargetCounter'] += 1
        self.tracer_table.iloc[:]['deathCounter'] += 1

        '''
        Step 1: Update status(after collector)
        Considering the calculation noise, the determination time 
        is made up when the target behavior is detected again.
        
        The target state is considered to be left only if the 
        target behavior is not performed continuously.
        '''
        for obj, b_result_column in zip(inputs, zip(*b_result)):
            pid = obj.track_id
            if any(b_result_column) and pid in self.analyst_table.index:
                if isnan(self.analyst_table.loc[pid, 'targetCounter']):
                    self.analyst_table.loc[pid, 'targetCounter'] = 0
                    self.analyst_table.loc[pid, 'nonTargetCounter'] = 0
                else:
                    self.analyst_table.loc[pid, 'targetCounter'] += self.analyst_table.loc[pid, 'nonTargetCounter']
                    self.analyst_table.loc[pid, 'nonTargetCounter'] = 0
            if pid in self.tracer_table.index:
                self.tracer_table.loc[pid, 'deathCounter'] = 0

        '''
        Step 2: Update status(in collector)
        According to the trigger, the inputs will be filtered out
        of the target. They will be sent to the collector side.
        '''
        for obj, c_result_column in zip(inputs, zip(*c_result)):
            pid = obj.track_id
            if any(c_result_column):
                self.analyst_table.loc[pid] = [None, 0]

        '''
        Step 3: Emit targets that activates the trigger and reaches the threshold
        There are three emit conditions.
            1. Targets that never activate the triggers over time.
            2. Targets that activate the triggers, but not reach the thresholds.
            3. Targets that activate the triggers and reach the thresholds.
        '''
        self.tracer_table = self.tracer_table.loc[self.tracer_table['deathCounter'] < self.tracer_life]

        if self.analyst_table.size == 0:
            return [], [], [], c_result, b_result

        nonTarget = isnan(self.analyst_table['targetCounter']) & (self.analyst_table['nonTargetCounter'] >= self.nonTargetThreshold)
        nonTargetIds = self.analyst_table.loc[nonTarget].index
        for i in nonTargetIds: self.tracer_table.loc[i] = [1, 0]
        self.analyst_table = self.analyst_table.loc[~nonTarget]

        wrongTarget = ~isnan(self.analyst_table['targetCounter']) & (self.analyst_table['nonTargetCounter'] >= self.waitingThreshold)
        wrongTargetIds = self.analyst_table.loc[wrongTarget].index
        for i in wrongTargetIds: self.tracer_table.loc[i] = [2, 0]
        self.analyst_table = self.analyst_table.loc[~wrongTarget]

        correctTarget = self.analyst_table['targetCounter'] >= self.targetThreshold
        correctTargetIds = self.analyst_table.loc[correctTarget].index
        for i in correctTargetIds: self.tracer_table.loc[i] = [3, 0]
        self.analyst_table = self.analyst_table.loc[~correctTarget]

        return nonTargetIds, wrongTargetIds, correctTargetIds, c_result, b_result

    def __virtualFence(self, vertex_type, fence, relay_ext=None):
        '''
        We apply Point in Polygon Strategies to check whether
        a single vertex is in specified range or not.

        vertex_type = (0,0)->left top
                      (0,1)->left bottom
                      (1,1)->right bottom
                      (1,0)->right top
        '''
        relay = []
        if relay_ext is not None:
            buffer_zone = PolygonExtension(fence, relay_ext)
        def instanceFence(objs, filtered_result):
            nonlocal relay
            result = []
            for obj, tag in zip(objs, filtered_result):
                if tag:
                    p = {
                        'x':obj.tlwh[0]+vertex_type['x']*obj.tlwh[2], 
                        'y':obj.tlwh[1]+vertex_type['y']*obj.tlwh[3]
                    }
                    tag = windingNumber(p, fence) != 0
                if (
                    relay_ext is not None and 
                    not tag and
                    obj.track_id in relay
                ):
                    tag = windingNumber(p, buffer_zone) != 0
                result.append(tag)
            relay = [obj.track_id for tag in result if tag]
            return result
        return instanceFence

    def __edgeTrigger(self):
        pre_filtered_ids = []
        def instanceEdgeTrigger(objs, filtered_result):
            result = []
            current_filtered_ids = []
            nonlocal pre_filtered_ids
            for obj, tag in zip(objs, filtered_result):
                if not tag:
                    result.append(obj.track_id in pre_filtered_ids)
                else:
                    current_filtered_ids.append(obj.track_id)
                    result.append(False)
            pre_filtered_ids = current_filtered_ids
            return result
        return instanceEdgeTrigger

    def __checkAttr(self, thres):
        def instanceCheck(objs, filtered_result):
            result = []
            for obj, tag in zip(objs, filtered_result):
                if tag:
                    aspect_ratio = obj.tlwh[2] / obj.tlwh[3]
                    result.append(thres['min'] < aspect_ratio < thres['max'])
                else:
                    result.append(False)
            return result
        return instanceCheck

    def visualize(self, frame, online_persons):
        for target in online_persons:
            tID = target.track_id
            if tID not in self.tracer_table.index:
                continue
            tType = self.tracer_table.loc[tID, 'type']
            xmin, ymin, w, h = target.tlwh
            box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
            if tType == 1:
                color = (82, 4, 28)
                text = 'bad'
            elif tType == 2:
                color = (176, 21, 61)
                text = 'wrong'
            elif tType == 3:
                color = (255, 255, 255)
                text = 'OK'
            cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=5)
            cv2.putText(frame, f'{tID}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 5, color, thickness=3)

        for target in online_persons:
            tID = target.track_id
            if tID not in self.analyst_table.index:
                continue
            xmin, ymin, w, h = target.tlwh
            box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
            if isnan(self.analyst_table.loc[tID, 'targetCounter']):
                color = (0, 255, 255)
                text = f"dirty={self.analyst_table.loc[tID, 'nonTargetCounter']}"
            else:
                color = (255, 153, 255)
                text = f"clean={self.analyst_table.loc[tID, 'targetCounter']}"
            cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=3)
            cv2.putText(frame, f'{tID}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=3)

    def __parseTrigger(self, node_description):
        triggers = []
        triggers_desc = node_description['triggers']
        for trigger in triggers_desc:
            if trigger['type'] == 'Fence':
                relay_ext = None
                if 'relay_ext' in trigger['parameter']:
                    relay_ext = trigger['parameter']['relay_ext']
                triggers.append(
                    self.__virtualFence(
                        trigger['parameter']['vertex_type'], 
                        trigger['parameter']['position'],
                        relay_ext=relay_ext
                    )
                )
            elif trigger['type'] == 'Edge':
                triggers.append(
                    self.__edgeTrigger()
                )
            elif trigger['type'] == 'AttrFilter':
                triggers.append(
                    self.__checkAttr(
                        trigger['parameter']['range']
                    )
                )
        return TriggerNode(triggers)

    def deserialize(self, trigger_description):
        collector_description = trigger_description['collector']
        base_description = trigger_description['base']
        for c_node_description in collector_description:
            self.collector.append(self.__parseTrigger(c_node_description))
        for b_node_description in base_description:
            self.base.append(self.__parseTrigger(b_node_description))

def windingNumber(point, vertexs):
    # https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
    def isLeft(p0, p1, p2):
        return ( (p1['x']-p0['x']) * (p2['y']-p0['y']) - (p2['x']-p0['x'])  *   (p1['y']-p0['y']) )
    wn = 0
    for i in range(len(vertexs)):
        if vertexs[i]['y'] <= point['y']:
            if vertexs[(i+1)%len(vertexs)]['y'] > point['y']:
                if isLeft(vertexs[i], vertexs[(i+1)%len(vertexs)], point) > 0:
                    wn += 1
        else:
            if vertexs[(i+1)%len(vertexs)]['y'] <= point['y']:
                if isLeft(vertexs[i], vertexs[(i+1)%len(vertexs)], point) < 0:
                    wn -= 1
    return wn