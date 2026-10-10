from flow import Chart

OUT = {}

# =====================================================================
# C1  OVERVIEW — swimlane
# =====================================================================
def overview():
    X = dict(BE=105, CS=325, DE=620, RD=940, OC=1205, HB=1470, MN=1735)
    lanes = [('Backend', X['BE'], 210), ('camera_stream + MediaMTX', X['CS'], 230), ('plate_detector', X['DE'], 360),
             ('Redis', X['RD'], 260), ('plate_ocr (3 workers)', X['OC'], 270), ('control_hub', X['HB'], 260), ('MinIO', X['MN'], 170)]
    c = Chart(1830, 2000, lanes)
    n = c.n
    DE, RD, OC, HB, BE, CS, MN = (X[k] for k in ('DE', 'RD', 'OC', 'HB', 'BE', 'CS', 'MN'))
    n('be1', BE, 110, 'Save camera\nHSET cameras:config\nPUBLISH config:updated', 'start')
    n('cs1', CS, 110, 'Register in MediaMTX\nmonitor online / offline\n(offline held 10 s)', 'proc')
    n('be2', BE, 290, 'Activate camera\nLPUSH cmd:ai:request', 'start')
    n('rd_cmd', RD, 290, 'cmd:ai:request', 'store', minw=170)
    n('de1', DE, 380, 'Command worker (BRPOP 2 s)\nwrite active_cameras ledger\nattach camera: RTSP reader,\ntracker, model instance', 'proc')
    n('de2', DE, 540, 'Newest frame → crop ROI\n→ YOLO detect → BYTETracker\n(auto-degrade if over capacity)', 'proc')
    n('de3', DE, 690, 'Trigger fired or\nhub asked again?', 'dec')
    n('de4', DE, 840, 'Build task: ≤5 crops + 1 frame\n(JPEG) · emit trigger / submitted', 'proc')
    n('de5', DE, 990, 'Track unseen\n> 30 frames?', 'dec')
    n('de6', DE, 1130, 'leave_scene task (if rules)\nemit track_ended · forget track', 'proc')
    n('rd_ev', RD, 840, 'hub:events (STREAM)', 'store', minw=190)
    n('rd_tasks', RD, 1070, 'ocr:tasks (LIST)', 'store', minw=190)
    n('oc1', OC, 1070, 'BRPOP task (1 s)\nvote class: car / motorcycle', 'good')
    n('oc2', OC, 1185, 'OCR every crop\nvote characters', 'good')
    n('oc3', OC, 1320, 'Valid?\nconf ≥ 0.70, 8 chars,\nplate pattern', 'dec')
    n('oc4', OC, 1465, 'Upload crop + best frame\n(2 PNG) to MinIO', 'good')
    n('oc5', OC, 1565, 'Answer ALWAYS sent\n(ok / invalid / skipped / error)', 'good')
    n('mn', MN, 1465, 'MinIO\nprivate bucket', 'store', minw=130)
    n('rd_res', RD, 1565, 'hub:results (STREAM)', 'store', minw=190)
    n('hb1', HB, 840, 'Read events\n(track_started, submitted,\ntrigger, track_ended)', 'hub')
    n('hb_p', HB, 960, 'Every 3 s (periodic cameras):\nask detector to re-read if\nnot satisfied, nothing in flight', 'hub')
    n('rd_ctl', RD, 960, 'hub:ctl:<engine> (LIST)', 'store', minw=190)
    n('hb2', HB, 1100, 'Trigger fired:\nresult known or\nsatisfied?', 'dec')
    n('hb3', HB, 1250, 'Defer trigger\nwait ≤ 15 s (task queued)\nor ≤ 3 s (no task)', 'hub')
    n('hb4', HB, 1565, 'Result arrives:\nvote (sum of confidences)\nsatisfied if ≥ 0.85 or 3 agree', 'hub')
    n('hb5', HB, 1700, 'Publish waiting triggers\nsend ack to detector', 'hub')
    n('hb6', HB, 1835, 'Track ended and all tasks\nanswered (or 30 s)?', 'dec')
    n('hb7', HB, 1975, 'Publish final\n(gate: ≥8 frames, ≥1 crop)', 'hub')
    n('rd_out', RD, 1975, 'plate:vehicle:results', 'store', minw=190)
    n('be3', BE, 1975, 'Backend reads\nplate record + images', 'end')

    c.e('be1', 'cs1'); c.e('be1', 'be2')
    c.e('cs1', 'de1', sa='b', ta='l', st='d', label='camera:events\n(online / offline)', pos=60)
    c.e('be2', 'rd_cmd', st='d', label='activated')
    c.e('rd_cmd', 'de1', sa='b', ta='r', st='d')
    c.e('de1', 'de2'); c.e('de2', 'de3')
    c.e('de3', 'de4', 'yes', st='y')
    c.e('de3', 'de2', 'no: next frame', sa='l', ta='l', via=[(DE - 215, 690), (DE - 215, 540)], st='l')
    c.e('de4', 'de5')
    c.e('de4', 'rd_ev', sa='r', ta='l', st='d', via=[(DE + 150, 840)])
    c.e('de4', 'rd_tasks', sa='r', ta='l', st='d', via=[(DE + 150, 840), (DE + 150, 1070)])
    c.e('de5', 'de6', 'yes', st='y')
    c.e('de5', 'de2', 'no', sa='l', ta='l', via=[(DE - 255, 990), (DE - 255, 540)], st='l')
    c.e('de6', 'rd_ev', sa='r', ta='l', st='d', via=[(DE + 150, 1130), (DE + 150, 840)], label='track_ended', pos=40)
    c.e('rd_tasks', 'oc1', st='d')
    c.e('oc1', 'oc2'); c.e('oc2', 'oc3')
    c.e('oc3', 'oc4', 'valid → ok · invalid → invalid', st='y', pos=40)
    c.e('oc4', 'oc5')
    c.e('oc4', 'mn', st='n', sa='r', ta='l', label='2 PNG', pos=90)
    c.e('oc5', 'rd_res', st='d', sa='l', ta='r')
    c.e('rd_ev', 'hb1', st='d', sa='r', ta='l')
    c.e('hb1', 'hb_p'); c.e('hb_p', 'hb2')
    c.e('hb_p', 'rd_ctl', sa='l', ta='r', st='d', label='request', pos=30)
    c.e('hb2', 'hb3', 'no', st='x')
    c.e('hb2', 'hb4', 'yes: publish now', st='y', sa='r', ta='r', via=[(HB + 150, 1100), (HB + 150, 1565)], pos=40)
    c.e('hb3', 'hb4')
    c.e('rd_res', 'hb4', st='d', sa='r', ta='l')
    c.e('hb4', 'hb5')
    c.e('hb5', 'rd_ctl', st='d', sa='l', ta='b', via=[(HB - 150, 1700), (RD, 1700)], label='ack + satisfied', pos=70)
    c.e('hb5', 'hb6')
    c.e('hb6', 'hb7', 'yes', st='y')
    c.e('hb6', 'hb1', 'no: keep waiting', sa='r', ta='r', via=[(HB + 185, 1835), (HB + 185, 840)], st='l', pos=60)
    c.e('hb7', 'rd_out', st='d', sa='l', ta='r')
    c.e('rd_out', 'be3', st='d', sa='l', ta='r', label='RPUSH', pos=200)
    c.e('mn', 'be3', sa='b', ta='b', via=[(MN, 2065), (BE, 2065)], st='n', label='backend loads images by path', pos=900)
    c.e('rd_ctl', 'de3', sa='l', ta='r', st='l', via=[(DE + 190, 960), (DE + 190, 690)], label='ack frees the lock;\nsatisfied stops OCR', pos=250)
    c.h = 2115
    return c

OUT['overview'] = overview()


# =====================================================================
# C2  DETECTOR ENGINE LOOP
# =====================================================================
def frameloop():
    c = Chart(1250, 100)
    X = 430
    ids = c.stack(X, 30, [
        ('f0', 'start', 'Engine loop — one iteration'),
        ('f1', 'proc', 'Apply add / remove camera commands\nDrain hub:ctl (result · request · state)'),
        ('f2', 'proc', 'For each camera: take the newest frame\n(reader thread keeps only the latest one)'),
        ('f3', 'dec', 'New frame since\nlast time?'),
        ('f4', 'proc', 'missed = new frames − 1 · fid += new frames\nCrop the camera ROI (keep offset → full-frame px)'),
        ('f5', 'dec', 'Due for detection?\nfid ≥ _next_due'),
        ('f6', 'proc', 'ONE inference call for all due frames\nGPU: one batched predict() · CPU: one instance per camera\nconf ≥ 0.25 · NMS IoU 0.7 · ≤ 300 boxes'),
        ('f7', 'proc', 'BYTETracker.update(detections, dt)\ndt = real camera frames since the last update'),
        ('f8', 'dec', 'Active track\nis new?'),
        ('f9', 'proc', 'Cut plate crop from ROI frame · sharpness (Laplacian)\nRun triggers  (flowchart 3)'),
        ('f10', 'proc', 'Update best-5 crops (rank = conf×100, area)\nUpdate best frame when the best crop improved'),
        ('f11', 'dec', 'Recognition\nrequested?'),
        ('f12', 'proc', 'Emit track_update (at most every 5 s)'),
        ('f13', 'dec', 'Any track unseen\n> 30 frames?'),
        ('f14', 'end', 'Debug video · [PERF] every 10 s · [STATS] every 50 loops\n→ next iteration'),
    ], gap=30)
    for a, b in zip(ids, ids[1:]):
        lab = {'f3': 'yes', 'f5': 'yes', 'f8': 'no', 'f11': 'no', 'f13': 'no'}.get(a, '')
        c.e(a, b, lab, st='y' if lab == 'yes' else ('x' if lab == 'no' else 'n'))
    n = c.nodes
    c.n('s_skip', X + 400, n['f3']['y'], 'skip this camera\n(nothing new)', 'ext')
    c.e('f3', 's_skip', 'no', st='x')
    c.n('s_coast', X + 400, n['f5']['y'], 'COAST: no inference, tracker not updated\n(counted as coasted, not missed)', 'ext')
    c.e('f5', 's_coast', 'no', st='x')
    c.n('s_new', X + 420, n['f8']['y'], 'uid = camera-engine-random10\nemit track_started {trigger flags}\nnew TrackRecState', 'proc')
    c.e('f8', 's_new', 'yes', st='y')
    c.e('s_new', 'f9', sa='b', ta='r', via=[(X + 420, n['f9']['y'])], st='n')
    c.n('s_sub', X + 420, n['f11']['y'], 'Try to submit crops\n(submit gate — flowchart 4)', 'proc')
    c.e('f11', 's_sub', 'yes', st='y')
    c.e('s_sub', 'f12', sa='b', ta='r', via=[(X + 420, n['f12']['y'])], st='n')
    c.n('s_end', X + 420, n['f13']['y'], 'End the track\n(flowchart 4)', 'proc')
    c.e('f13', 's_end', 'yes', st='y')
    c.e('s_end', 'f14', sa='b', ta='r', via=[(X + 420, n['f14']['y'])], st='n')
    # loops
    c.e('s_skip', 'f13', sa='b', ta='r', via=[(X + 400, n['f13']['y'] - 70)] if False else [(X + 560, n['f3']['y']), (X + 560, n['f13']['y'])], st='l', label='next camera', pos=40) if False else None
    c.e('f14', 'f1', sa='l', ta='l', via=[(X - 330, n['f14']['y']), (X - 330, n['f1']['y'])], st='l', label='repeat', pos=60)
    # coast / skip rejoin the camera loop
    c.e('s_skip', 'f2', sa='t', ta='r', via=[(X + 400, n['f2']['y'])], st='l', label='next camera', pos=18)
    c.e('s_coast', 'f2', sa='r', ta='r', via=[(X + 560, n['f5']['y']), (X + 560, n['f2']['y'])], st='l')
    return c.fit()


# =====================================================================
# C3  TRIGGERS
# =====================================================================
def triggers():
    c = Chart(1100, 100)
    X = 330
    ids = c.stack(X, 30, [
        ('t0', 'start', 'Per track, per detected frame\nanchor = bottom-centre of the plate box (full-frame px)'),
        ('t1', 'dec', 'Line set, track age ≥ 5 frames\nand avg conf ≥ 0.40?'),
        ('t2', 'dec', 'Anchor changed side\nof the line?'),
        ('t3', 'dec', 'Cooldown (30 frames)\nsince last crossing over?'),
        ('t4', 'good', 'EVENT cross_line {direction, point, conf}'),
        ('t5', 'dec', 'Stop ROI set, anchor inside polygon\nand avg conf ≥ 0.30?'),
        ('t6', 'proc', 'inside_frames += 1\n≥ 2 frames → entry confirmed, timer starts'),
        ('t7', 'dec', '≥ 6 position samples and\n≥ 2.0 s since entry?'),
        ('t8', 'dec', 'Net movement over the last 1.5 s\n≤ 20 px/s?'),
        ('t9', 'good', 'EVENT stop_roi {duration, velocity, point}'),
        ('t10', 'end', 'Engine: once per event name per track\n→ hub:events trigger + request a recognition'),
    ], gap=30)
    lab = {'t1': ('yes', 'y'), 't2': ('yes', 'y'), 't3': ('yes', 'y'), 't5': ('yes', 'y'), 't7': ('yes', 'y'), 't8': ('yes', 'y')}
    for a, b in zip(ids, ids[1:]):
        if a == 't4':
            c.e('t4', 't5'); continue
        if a == 't9':
            c.e('t9', 't10'); continue
        l, st = lab.get(a, ('', 'n'))
        c.e(a, b, l, st=st)
    # the 'no' exits
    c.e('t1', None, 'no', sa='r', st='x', stub=210)
    c.e('t2', None, 'no (same side)', sa='r', st='x', stub=210)
    c.e('t3', None, 'no (too soon)', sa='r', st='x', stub=210)
    c.e('t5', None, 'no: inside counter reset;\nconfirmed → ROI exit', sa='r', st='x', stub=210)
    c.e('t7', None, 'no: keep sampling', sa='r', st='x', stub=210)
    c.e('t8', None, 'no: still moving', sa='r', st='x', stub=210)
    # t4 also continues to t5 (both checks run each frame) - already drawn
    c.n('side', 980, c.nodes['t5']['y'], 'Both checks run on every frame.\nA track can fire cross_line and stop_roi\n(each once).', 'ext')
    return c.fit()


# =====================================================================
# C4  SUBMIT GATE + END TRACK
# =====================================================================
def submit_end():
    c = Chart(1300, 100)
    X1, X2 = 300, 900
    ids = c.stack(X1, 30, [
        ('s0', 'start', 'Something asks for a read:\ntrigger fired (local) · hub periodic request · track ends'),
        ('s1', 'dec', 'Hub already said\n"satisfied"?'),
        ('s2', 'dec', 'A task of this track\nis in flight?'),
        ('s3', 'dec', 'Any crop stored\nfor this track?'),
        ('s4', 'dec', 'Crop set changed since\nthe last send?\n(best frame no. : crop count)'),
        ('s5', 'proc', 'Pick the stage by priority\nleave_scene 3 > cross_line / stop_roi 2 > periodic 1'),
        ('s6', 'proc', 'Encode ≤ 5 crops + 1 best frame as JPEG\n+ camera, track, uid, stage, class flags, scores\n→ pickle'),
        ('s7', 'store', 'emit "submitted" → hub:events, THEN LPUSH task → ocr:tasks\n(in-memory outbox keeps the order; retry 0.2 → 5 s)'),
    ], gap=30)
    c.e('s0', 's1')
    c.e('s1', 's2', 'no', st='x')
    c.e('s2', 's3', 'no', st='x')
    c.e('s3', 's4', 'yes', st='y')
    c.e('s4', 's5', 'yes', st='y')
    c.e('s5', 's6'); c.e('s6', 's7', st='d')
    c.e('s1', None, 'yes: nothing sent', st='y', stub=210)
    c.e('s2', None, 'yes: wait for the hub ack\n(or 15 s SUBMIT_TIMEOUT)', st='y', stub=210)
    c.e('s3', None, 'no: nothing to send yet', st='x', stub=210)
    c.e('s4', None, 'no: wait for a better crop;\nrequest stays pending', st='x', stub=210)
    ids2 = c.stack(X2, 30, [
        ('e0', 'start', 'Track unseen for > 30 camera frames\n(or camera removed · engine stopping)'),
        ('e1', 'dec', 'leave_scene on, hub not satisfied,\nseen ≥ 8 frames, ≥ 1 crop?'),
        ('e2', 'proc', 'Build leave_scene task\n(same path as the submit gate)'),
        ('e3', 'store', 'emit track_ended {reason, seen_frames, n_crops,\nduration, finalize_task_id}'),
        ('e4', 'end', 'Forget crops, best frame,\ntrigger state, recognition state'),
    ], gap=30)
    c.e('e0', 'e1'); c.e('e1', 'e2', 'yes', st='y'); c.e('e2', 'e3', st='d')
    c.e('e1', 'e3', 'no: no final task', sa='r', ta='r', via=[(X2 + 210, c.nodes['e1']['y']), (X2 + 210, c.nodes['e3']['y'])], st='x', pos=30)
    c.e('e3', 'e4')
    return c.fit()


# =====================================================================
# C5  OCR
# =====================================================================
def ocr():
    c = Chart(1250, 100)
    X = 420
    ids = c.stack(X, 30, [
        ('o0', 'start', 'BRPOP internal:ocr:tasks (1 s)\nonly while demand = processing'),
        ('o1', 'dec', 'Has task_id, camera_id,\ntrack_id, trigger_type?'),
        ('o2', 'dec', 'Crops present and\nat least one decodes?'),
        ('o3', 'proc', 'Class vote: majority of the crops\' class flags'),
        ('o4', 'dec', 'Voted class?'),
    ], gap=34)
    c.e('o0', 'o1'); c.e('o1', 'o2', 'yes', st='y'); c.e('o2', 'o3', 'yes', st='y'); c.e('o3', 'o4')
    y4 = c.nodes['o4']['y']
    c.n('car', X - 230, y4 + 130, 'CAR (class 0), per crop:\nCLAHE on V channel → resize 256×64\n→ PaddleOCR recognition only\nvote character by character\n(only 8-char candidates)', 'good')
    c.n('mot', X + 230, y4 + 130, 'MOTORCYCLE (class 1), per crop:\nresize 240×200 → detect + recognise\ndrop boxes < 5 % of the largest\nvote each part', 'good')
    c.e('o4', 'car', 'car (0)', sa='l', ta='t', st='y', via=[(X - 230, y4)])
    c.e('o4', 'mot', 'motorcycle (1)', sa='r', ta='t', st='y', via=[(X + 230, y4)])
    c.n('val', X, y4 + 290, 'Valid?\ncar: conf ≥ 0.70, 8 chars,\n2 digits + letter + 3 + 2\nmotorcycle: format rules', 'dec')
    c.e('car', 'val', sa='b', ta='l', st='n', via=[(X - 230, y4 + 290)])
    c.e('mot', 'val', sa='b', ta='r', st='n', via=[(X + 230, y4 + 290)])
    c.n('up', X, y4 + 440, 'Upload crop[0] + best frame as PNG to MinIO\nvalid → dynamics/ValidPlateImage/vp_… + ValidVehicleImage/vf_…\ninvalid → InvalidPlateImage/ip_… + InvalidVehicleImage/if_…', 'good')
    c.e('val', 'up', 'yes → status ok\nno → status invalid + reason', st='y', pos=30)
    c.n('push', X, y4 + 560, 'XADD internal:hub:results {uid, task_id, stage, status,\nplate, confidence, validity, image paths}\n8 attempts, 0.2 → 5 s backoff (~25 s)', 'store')
    c.e('up', 'push', st='d')
    c.n('done', X, y4 + 670, '[TASK-DONE] total_processing_ms · [OCR-STATS] every 30 s → next task', 'end')
    c.e('push', 'done')
    # error / skip exits go to push
    c.n('err', X + 520, c.nodes['o1']['y'], 'status = error\n"invalid OCR task contract"', 'warn' if False else 'end')
    c.e('o1', 'err', 'no', st='x')
    c.n('skip', X + 520, c.nodes['o2']['y'], 'status = skipped\n(no crops / all decodes failed)', 'end')
    c.e('o2', 'skip', 'no', st='x')
    c.e('err', 'push', sa='r', ta='r', st='x', via=[(X + 760, c.nodes['err']['y']), (X + 760, c.nodes['push']['y'])])
    c.e('skip', 'push', sa='b', ta='r', st='x', via=[(X + 520, c.nodes['push']['y'])], label='always answer the hub', pos=250)
    c.e('skip', 'err', sa='b', ta='t', st='x', pos=1) if False else None
    c.n('unk', X + 560, y4, 'other class → status = error', 'end') if False else None
    return c.fit()


# =====================================================================
# C6  HUB — events
# =====================================================================
def hub_events():
    c = Chart(2200, 100)
    n = c.n
    n('h0', 1000, 40, 'XREADGROUP events + results (block 200 ms, up to 200 entries)\nunacknowledged entries are read again after a restart', 'start')
    n('h1', 1000, 140, 'core.handle(kind)', 'dec', minw=230)
    c.e('h0', 'h1')
    BUS = 215
    cols = {'engine_started': 110, 'track_started': 330, 'track_update': 545, 'submitted': 770, 'track_ended': 995, 'trigger': 1290, 'result': 1700}
    Y = 290
    n('k_engine_started', cols['engine_started'], Y, 'End every live track of\nthat engine with an older\nboot_id (engine_restarted)', 'hub', minw=190)
    n('k_track_started', cols['track_started'], Y, 'Create track state\nstore trigger flags\nperiodic due at +1 s', 'hub', minw=190)
    n('k_track_update', cols['track_update'], Y, 'Absorb frame / crop counts\nreset the stale timer\nre-check "satisfied"', 'hub', minw=190)
    n('k_submitted', cols['submitted'], Y, 'Mark task_id in flight;\ntriggers waiting without\na task now wait for it', 'hub', minw=190)
    n('k_track_ended', cols['track_ended'], Y, 'Mark ended; remember\nfinalize_task_id in flight\n→ try to close (flowchart 7)', 'hub', minw=190)
    n('tr1', cols['trigger'], Y, 'Known event, enabled on\nthis camera, first time\nfor this track?', 'dec')
    n('rs1', cols['result'], Y, 'Duplicate task_id?', 'dec')
    for k in ('engine_started', 'track_started', 'track_update', 'submitted', 'track_ended', 'trigger', 'result'):
        tgt = {'trigger': 'tr1', 'result': 'rs1'}.get(k, 'k_' + k)
        x = cols[k]
        c.e('h1', tgt, k, sa='b', ta='t', via=[(1000, BUS), (x, BUS)], pos='end')
    # trigger subtree
    n('tr2', cols['trigger'], 450, 'Hub already satisfied,\nor its task already answered?', 'dec')
    c.e('tr1', 'tr2', 'yes', st='y')
    c.e('tr1', None, 'no: ignore', st='x', sa='r', stub=120)
    n('tr3', 1150, 610, 'PUBLISH now\n(immediate / after_recognition)', 'good')
    n('tr4', 1430, 640, 'DEFER the trigger:\nwait ≤ 15 s if its task is queued\n≤ 3 s if no task could be sent\n(timers: flowchart 7)', 'hub')
    c.e('tr2', 'tr3', 'yes', st='y', sa='b', ta='t', via=[(cols['trigger'], 530), (1150, 530)], pos='end')
    c.e('tr2', 'tr4', 'no', st='x', sa='b', ta='t', via=[(cols['trigger'], 530), (1430, 530)], pos='end')
    # result subtree
    n('rs2', cols['result'], 450, 'Track already CLOSED?', 'dec')
    c.e('rs1', 'rs2', 'no', st='x')
    c.e('rs1', None, 'yes: ignore', st='y', sa='r', stub=120)
    n('rs3', 2020, 600, 'LATE result:\nanswer changed? → publish the\nfinal again (revision + 1)\nelse log and drop', 'ext')
    n('rs4', 1760, 730, 'Add to the vote (valid only, sum of confidences)\nre-check satisfied (≥ 0.85 or 3 agree)\nsend ctl ack to the detector\npublish deferred triggers waiting for this task', 'hub')
    c.e('rs2', 'rs3', 'yes', st='y', sa='r', ta='t', via=[(2020, 450)])
    c.e('rs2', 'rs4', 'no', st='x')
    # apply
    n('ap', 1000, 960, 'Apply effects:\nRPUSH plate:vehicle:results (trigger / final records)\nLPUSH hub:ctl:<engine> (ack, satisfied, request) · EXPIRE 120 s\nSET hub:track:<uid> checkpoints (TTL 3600 s) · DEL expired ones', 'store')
    n('ack', 1000, 1100, 'XACK the batch → next read', 'end')
    c.e('ap', 'ack')
    BUS2 = 760
    for k, x in (('k_engine_started', cols['engine_started']), ('k_track_started', cols['track_started']),
                 ('k_track_update', cols['track_update']), ('k_submitted', cols['submitted']), ('k_track_ended', cols['track_ended'])):
        c.e(k, 'ap', sa='b', ta='t', via=[(x, BUS2), (1000, BUS2)])
    c.e('tr3', 'ap', sa='b', ta='t', via=[(1150, BUS2), (1000, BUS2)])
    c.e('tr4', 'ap', sa='b', ta='t', via=[(1430, BUS2), (1000, BUS2)])
    c.e('rs4', 'ap', sa='b', ta='t', via=[(1760, 880), (1000, 880)])
    c.e('rs3', 'ap', sa='b', ta='t', via=[(2020, BUS2 + 40), (1000, BUS2 + 40)])
    return c.fit()


# =====================================================================
# C7  HUB — timers and closing a track
# =====================================================================
def hub_timers():
    c = Chart(1800, 100)
    X = 330
    ids = c.stack(X, 30, [
        ('k0', 'start', 'Tick every 0.2 s — for every track in memory'),
        ('k1', 'dec', 'Track CLOSED?'),
        ('k2', 'dec', 'Track ENDED?'),
        ('k3', 'dec', 'No event at all for\n≥ 120 s (stale)?'),
        ('k4', 'dec', 'A deferred trigger waited ≥ its limit?\n15 s (its task is queued) · 3 s (no task)'),
        ('k5', 'dec', 'Periodic due?\n(camera has the periodic flag;\nfirst at +1 s, then every 3 s)'),
        ('k6', 'dec', 'A task in flight\nfor ≥ 90 s?'),
        ('k7', 'end', 'next track'),
    ], gap=34)
    lab = {'k1': 'no', 'k2': 'no', 'k3': 'no', 'k4': 'no', 'k5': 'no', 'k6': 'no'}
    for a, b in zip(ids, ids[1:]):
        c.e(a, b, lab.get(a, ''), st='x' if a in lab else 'n')
    y = lambda i: c.nodes[i]['y']
    c.n('r1', 780, y('k1'), 'closed ≥ 300 s ago?\nyes → delete track + checkpoint', 'proc')
    c.e('k1', 'r1', 'yes', st='y')
    c.e('r1', 'k7', sa='b', ta='r', via=[(780, y('k7'))], st='l', pos=1) if False else None
    c.n('r2', 780, y('k2'), 'TRY TO CLOSE  (flowchart on the right)', 'proc')
    c.e('k2', 'r2', 'yes', st='y')
    c.n('r3', 780, y('k3'), 'END the track itself\nreason = stale → try to close', 'proc')
    c.e('k3', 'r3', 'yes', st='y')
    c.n('r4', 780, y('k4'), 'PUBLISH that trigger now\n(resolution = timeout)', 'good')
    c.e('k4', 'r4', 'yes', st='y')
    c.n('r5', 780, y('k5'), 'Only if not satisfied and nothing in flight:\nLPUSH hub:ctl {request, stage: periodic}\n→ detector re-sends crops if a better set exists', 'proc')
    c.e('k5', 'r5', 'yes', st='y')
    c.n('r6', 780, y('k6'), 'Forget that task\n(so it cannot block periodic reads)', 'proc')
    c.e('k6', 'r6', 'yes', st='y')
    # close chain
    X2 = 1330
    c.stack(X2, 30, [
        ('c0', 'start', 'Try to close an ended track'),
        ('c1', 'dec', 'Tasks in flight and\n< 30 s since the end?'),
        ('c2', 'proc', 'Publish triggers still deferred\n(resolution = track_ended)\nIf it timed out: unanswered tasks → missing_tasks'),
        ('c3', 'dec', 'seen ≥ 8 frames and ≥ 1 crop,\nor something already published?'),
        ('c4', 'good', 'PUBLISH FINAL  update_type = leave_scene\nis_final, revision 1, complete / missing_tasks'),
        ('c5', 'store', 'closed — kept 300 s for late results'),
    ], gap=32)
    c.e('c0', 'c1'); c.e('c1', 'c2', 'no', st='x'); c.e('c2', 'c3'); c.e('c3', 'c4', 'yes', st='y'); c.e('c4', 'c5')
    c.e('c1', None, 'yes: keep waiting', st='y', sa='r', stub=140)
    c.n('c6', X2 + 370, c.nodes['c4']['y'], 'DROP the track\n(backend never hears of it)', 'end')
    c.e('c3', 'c6', 'no', st='x', sa='r', ta='t', via=[(X2 + 370, c.nodes['c3']['y'])])
    c.e('r2', 'c0', sa='r', ta='l', st='d', via=[(1020, y('k2')), (1020, 47), ], label='', pos=0) if False else None
    c.e('r2', 'c0', sa='r', ta='t', st='d', via=[(1000, y('k2')), (1000, 12), (X2, 12)], label='')
    c.e('r3', 'c0', sa='r', ta='t', st='d', via=[(1020, y('k3')), (1020, 12), (X2, 12)]) if False else None
    return c.fit()


# =====================================================================
# C8  CAMERA — camera_stream, commands, events, errors
# =====================================================================
def camera():
    c = Chart(2150, 100)
    n = c.n
    # camera_stream column
    XA = 250
    c.stack(XA, 30, [
        ('a0', 'start', 'camera_stream\nbackend wrote cameras:config\nand published config:updated'),
        ('a1', 'proc', 'sync_cameras → cameras:details\n(connected = false, "Waiting for connection")'),
        ('a2', 'dec', 'Camera answers on its\nRTSP port? (TCP ping 0.2 s)'),
        ('a3', 'proc', 'Register in MediaMTX\n(TCP, sourceOnDemand = false, 1 s timeout)'),
        ('a4', 'proc', 'Poll MediaMTX /paths/list\nevery 0.5 s (1.5 s timeout)'),
        ('a5', 'dec', 'Path "ready"?'),
        ('a6', 'dec', 'Offline for\n≥ 10 s (hold)?'),
        ('a7', 'store', 'cameras:details + PUBLISH camera:events\n(connected true / false, error)'),
    ], gap=30)
    c.e('a0', 'a1'); c.e('a1', 'a2'); c.e('a2', 'a3', 'yes', st='y'); c.e('a3', 'a4'); c.e('a4', 'a5')
    c.e('a2', None, 'no: try again next loop (0.3 s)', st='x', sa='l', stub=1) if False else None
    c.e('a5', 'a7', 'yes: online at once', st='y', sa='l', ta='l', via=[(XA - 200, c.nodes['a5']['y']), (XA - 200, c.nodes['a7']['y'])], pos=60)
    c.e('a5', 'a6', 'no', st='x')
    c.e('a6', 'a7', 'yes: publish offline', st='y')
    c.e('a6', None, 'no: keep holding', st='x', sa='r', stub=150)
    # commands column
    XB = 800
    c.stack(XB, 30, [
        ('b0', 'start', 'Backend: LPUSH cmd:ai:request\n{action, camera_id, request_id}'),
        ('b1', 'proc', 'Detector command worker\nBRPOP cmd:ai:request (2 s)'),
        ('b2', 'dec', 'action?'),
        ('b3', 'proc', 'mark_active in the durable ledger (with config) FIRST\nnotify demand → OCR starts popping tasks'),
        ('b4', 'proc', 'cancel pending offline timer · reset error counter'),
        ('b5', 'dec', 'Already running?'),
        ('b6', 'dec', 'Camera online now?\n(details.connected)'),
        ('b7', 'proc', 'ATTACH: engine with fewest cameras (new one if full)\nRTSP reader thread · BYTETracker · model instance\ncapacity check (✅ / 🚨) · cadence'),
        ('b8', 'store', 'cameras:<id>:ai_status = running\ncmd:ai:response:<request_id> = OK (TTL 60 s)'),
    ], gap=30)
    for a, b in zip(['b0', 'b1', 'b2'], ['b1', 'b2', 'b3']):
        c.e(a, b, 'activated' if a == 'b2' else '', st='y' if a == 'b2' else 'n')
    c.e('b3', 'b4'); c.e('b4', 'b5'); c.e('b5', 'b6', 'no', st='x'); c.e('b6', 'b7', 'yes', st='y'); c.e('b7', 'b8', st='d')
    c.e('b5', None, 'yes: answer OK', st='y', sa='r', stub=120)
    c.e('b6', None, 'no: wait for the online event\n(answer OK, attach later)', st='x', sa='r', stub=120)
    yb2 = c.nodes['b2']['y']
    c.n('d1', XB + 540, yb2 + 70, 'mark_inactive · notify demand\n(OCR stops when no camera is left)\ncancel timers · detach camera', 'proc')
    c.n('d2', XB + 540, yb2 + 190, 'ai_status = stopped_by_user\nresponse OK', 'store')
    c.e('b2', 'd1', 'deactivated', st='x', sa='r', ta='t', via=[(XB + 540, yb2)])
    c.e('d1', 'd2', st='d')
    # events and errors column
    XC = 1850
    c.stack(XC, 30, [
        ('c0', 'start', 'camera:events from camera_stream'),
        ('c1', 'dec', 'connected?'),
        ('c2', 'dec', 'Activated by the backend\nand not running?'),
        ('c3', 'proc', 'cancel offline timer → ATTACH'),
    ], gap=30)
    c.e('c0', 'c1'); c.e('c1', 'c2', 'online', st='y'); c.e('c2', 'c3', 'yes', st='y')
    c.e('c2', None, 'no: ignore', st='x', sa='l', stub=100)
    yc1 = c.nodes['c1']['y']
    c.stack(XC, 330, [
        ('c4', 'dec', 'Camera running?'),
        ('c5', 'proc', 'start 10 s offline timer\n(CAMERA_OFFLINE_GRACE)'),
        ('c6', 'dec', 'Back online within 10 s?'),
        ('c7', 'proc', 'DETACH · ai_status = stopped\n"Camera disconnected"\n(stays in the ledger)'),
    ], gap=30)
    c.e('c1', 'c4', 'offline', st='x', sa='r', ta='r', via=[(XC + 200, yc1), (XC + 200, c.nodes['c4']['y'])])
    c.e('c4', 'c5', 'yes', st='y'); c.e('c5', 'c6'); c.e('c6', 'c7', 'no', st='x')
    c.e('c6', None, 'yes: cancel the timer', st='y', sa='r', stub=100)
    c.e('c4', None, 'no: ignore', st='x', sa='l', stub=100)
    c.e('c7', 'a0', sa='b', ta='b', via=[(XC, 960), (XA, 960)], st='n') if False else None
    c.stack(XC, 760, [
        ('e0', 'start', 'Engine reports status = error\n(inference exception, resource shortage)'),
        ('e1', 'dec', 'Retries used < 1?\n(counter resets after 600 s)'),
        ('e2', 'proc', 'DETACH · wait 3 s'),
        ('e3', 'dec', 'Still activated?'),
        ('e4', 'good', 'ATTACH again\n(retry 1 of 1)'),
    ], gap=30)
    c.e('e0', 'e1'); c.e('e1', 'e2', 'yes', st='y'); c.e('e2', 'e3'); c.e('e3', 'e4', 'yes', st='y')
    c.e('e1', None, 'no: stay stopped', st='x', sa='l', stub=120)
    c.e('e3', None, 'no: leave it', st='x', sa='l', stub=100)
    # re-attach hints: events attach -> b7
    c.e('c3', 'b7', sa='l', ta='r', st='l', via=[(XC - 300, c.nodes['c3']['y']), (XC - 300, c.nodes['b7']['y'] + 0)], label='same attach', pos=40) if False else None
    return c.fit()


# =====================================================================
# C9  STARTUP
# =====================================================================
def startup():
    c = Chart(1750, 100)
    c.stack(280, 30, [
        ('d0', 'start', 'plate_detector starts'),
        ('d1', 'proc', 'Probe device: torch.cuda.is_available()\n(DETECTION_DEVICE gpu / cpu / auto)'),
        ('d1b', 'dec', 'Model files found?'),
        ('d2', 'proc', 'Wait for Redis (retry every 2 s)'),
        ('d3', 'proc', 'Start threads: command worker · camera events ·\nengine status · rebalance (30 s) · heartbeat (10 s, TTL 30 s)\nHealth server :8010'),
        ('d4', 'proc', 'Self-heal: read internal:detector:state\nstart_idle(n engines, default 1)'),
        ('d5', 'dec', 'Active-camera ledger\nnot empty?'),
        ('d6', 'good', 'start_process → reconcile:\nattach every active, online camera'),
        ('d7', 'proc', 'Each engine process: build backend (load model),\nCPU: capacity calibration (engine 0),\nwarm-up 10 runs, hub client → emit engine_started'),
        ('d8', 'end', 'Engine loop (flowchart 2)'),
    ], gap=26)
    for a, b in [('d0', 'd1'), ('d1', 'd1b'), ('d1b', 'd2'), ('d2', 'd3'), ('d3', 'd4'), ('d4', 'd5'), ('d5', 'd6'), ('d6', 'd7'), ('d7', 'd8')]:
        c.e(a, b, 'yes' if a in ('d1b', 'd5') else '', st='y' if a in ('d1b', 'd5') else 'n')
    c.e('d1b', None, 'no: CPU model → fall back to .pt on CPU;\n.pt missing → startup error', st='x', sa='r', stub=170)
    c.e('d5', 'd7', 'no: stay idle until a camera\nis activated', st='x', sa='r', ta='r', via=[(280 + 210, c.nodes['d5']['y']), (280 + 210, c.nodes['d7']['y'])], pos=60)
    c.stack(980, 30, [
        ('o0', 'start', 'plate_ocr starts'),
        ('o1', 'proc', 'Pool starts 3 worker processes\n(DEFAULT_WORKER_COUNT or saved count)'),
        ('o2', 'proc', 'Each worker: wait for Redis · load PaddleOCR\ncar + motorcycle models (≤ 180 s) · warm-up\nensure MinIO private bucket exists'),
        ('o3', 'dec', 'Pool in processing mode?'),
        ('o4', 'good', 'BRPOP ocr:tasks (flowchart 5)'),
    ], gap=30)
    c.e('o0', 'o1'); c.e('o1', 'o2'); c.e('o2', 'o3'); c.e('o3', 'o4', 'yes', st='y')
    c.e('o3', None, 'no: idle, poll every 0.2 s\n(demand event switches it on)', st='x', sa='r', stub=160)
    c.n('o5', 980, c.nodes['o4']['y'] + 100, 'Watchdog every 5 s: respawn a crashed worker', 'ext')
    c.e('o4', 'o5', st='l')
    c.stack(1480, 30, [
        ('h0', 'start', 'control_hub starts'),
        ('h1', 'proc', 'Wait for Redis (retry every 2 s)'),
        ('h2', 'dec', 'Got the leader lease?\n(TTL 15 s, renew every 5 s)'),
        ('h3', 'proc', 'Restore track checkpoints\n(hub:track:<uid>)'),
        ('h4', 'proc', 'Ensure consumer groups (new group starts at tail)\nre-read unacknowledged entries'),
        ('h5', 'good', 'Live loop (flowcharts 6 and 7)\nheartbeat every 10 s (TTL 30 s)'),
    ], gap=30)
    c.e('h0', 'h1'); c.e('h1', 'h2'); c.e('h2', 'h3', 'yes', st='y'); c.e('h3', 'h4'); c.e('h4', 'h5')
    c.e('h2', None, 'no: standby, retry\nevery ≤ 2 s', st='x', sa='l', stub=100)
    return c.fit()


for _n in ('overview', 'frameloop', 'triggers', 'submit_end', 'ocr', 'hub_events', 'hub_timers', 'camera', 'startup'):
    OUT[_n] = globals()[_n]()
