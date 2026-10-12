import os, re, charts
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')).read()
flowcss = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowcss.css')).read()

def sec(id_):
    m = re.search(r'<section id="%s">.*?</section>' % id_, src, re.S)
    return m.group(0)

head = src[:src.index('<nav>')]
head = head.replace('</style>', flowcss + '''
.chartbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:12px 0 4px;overflow:hidden}
.chartbar{display:flex;gap:6px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--line);background:var(--soft);font-size:12.5px;color:var(--muted)}
.chartbar b{color:var(--ink);font-size:13px;margin-right:auto}
.chartbar button{font-size:12px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:2px 9px;cursor:pointer}
.chartscroll{overflow:auto;max-height:86vh}
.key{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 0;font-size:12.5px;color:var(--muted);align-items:center}
.key i{display:inline-block;width:26px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.key .ky{border-color:var(--green)} .key .kn{border-color:var(--red)} .key .kd{border-color:var(--purple)} .key .kl{border-color:var(--amber);border-top-style:dashed} .key .kg{border-color:var(--muted)}
</style>''', 1)

def chart(name, title, caption=''):
    return (f'<div class="chartbox"><div class="chartbar"><b>{title}</b>'
            '<button data-z="out" type="button">−</button><button data-z="fit" type="button">Fit</button>'
            '<button data-z="one" type="button">100%</button><button data-z="in" type="button">+</button></div>'
            f'<div class="chartscroll">{charts.OUT[name].svg()}</div></div>'
            + (f'<p class="small" style="margin:4px 2px 0">{caption}</p>' if caption else ''))

def S(id_, no, title, lead, body):
    return f'<section id="{id_}"><h2><span class="no">{no}</span>{title}</h2><p class="lead">{lead}</p>{body}</section>'

nav = '''<nav><div class="in">
  <a href="#map">1 Services</a><a href="#overview">2 Whole pipeline</a><a href="#boot">3 Startup</a><a href="#camera">4 Cameras</a>
  <a href="#frame">5 Detector</a><a href="#ocr">6 OCR</a><a href="#hub">7 Control hub</a><a href="#seq">8 Sequence</a>
  <a href="#redis">9 Redis keys</a><a href="#images">10 Images</a><a href="#timeouts">11 Timeouts</a><a href="#findings">12 Findings</a>
  <button id="theme" type="button">Theme</button>
</div></nav>'''

key = ('<div class="key"><span><i class="ky"></i>yes</span><span><i class="kn"></i>no</span>'
       '<span><i class="kd"></i>data through Redis</span><span><i class="kl"></i>loop / feedback</span><span><i class="kg"></i>normal flow</span>'
       '<span>◇ decision</span><span>▭ step</span><span>⛁ Redis queue / store</span></div>')

sections = []
sections.append(sec('map').replace('<span class="no">1</span>', '<span class="no">1</span>'))
sections.append(S('overview', 2, 'Whole pipeline — flowchart 1',
  'Everything on one page, in swimlanes: camera registration, the detector loop, Redis queues, OCR, the control hub, and the record the backend reads. '
  'Use Fit / + / − to zoom; the detailed flowcharts below zoom into each part.', key + chart('overview', 'Flowchart 1 — whole pipeline')))
sections.append(S('boot', 3, 'Startup', 'What each service does when its container starts.', chart('startup', 'Startup — detector · OCR · control hub')))
sections.append(S('camera', 4, 'Camera lifecycle',
  'Left: camera_stream registers the camera and decides online / offline. Middle: the backend command (activate / deactivate). Right: online / offline events and the error restart.',
  chart('camera', 'Camera lifecycle — registration, commands, events, error restart')))
sections.append(S('frame', 5, 'Detector',
  'The engine loop, the trigger rules, and the gate that decides whether crops go to OCR.',
  chart('frameloop', 'Flowchart 2 — engine loop (one iteration)',
        'Auto-degrade / capacity (CPU): at startup engine 0 measures the loop time for 1…12 cameras (3 warm-up + 20 timed rounds, p95); budget = detect_every × 1000 / 25 fps × 0.7 = 28 ms at one camera. '
        'Above capacity, the per-camera detect interval is computed from those measurements (never below 1, never slower than 8 fps) and the skipped frames are "coasted". '
        'Engine placement: fewest cameras first; GPU 6 per engine, CPU 16 (one engine). Rebalance check every 30 s.')
  + chart('triggers', 'Flowchart 3 — triggers (cross_line, stop_roi)')
  + chart('submit_end', 'Flowchart 4 — submit gate (left) and ending a track (right)')))
sections.append(S('ocr', 6, 'OCR worker', 'One task = up to 5 crops of one track + 1 frame. A worker handles one task at a time; three workers run in parallel.',
  chart('ocr', 'Flowchart 5 — one OCR task')))
sections.append(S('hub', 7, 'Control hub',
  'The only component that decides what the backend receives. Per track uid (camera-engine-random10). '
  'Satisfied = best valid confidence ≥ 0.85, or ≥ 3 agreeing valid results with one candidate. The plate is chosen by a confidence-weighted vote over valid results only (sum of confidences per plate).',
  chart('hub_events', 'Flowchart 6 — events and results coming in')
  + chart('hub_timers', 'Flowchart 7 — timers and closing a track',
          'Backend records: update_type cross_line / stop_roi (once per event per track) and one final leave_scene (is_final, revision, complete, missing_tasks). Periodic re-reads are not published as records; they only feed the vote.')))

body = '\n'.join(sections) + '\n' + sec('seq').replace('<span class="no">5</span>', '<span class="no">8</span>') \
    + '\n' + sec('redis').replace('<span class="no">8</span>', '<span class="no">9</span>') \
    + '\n' + sec('images').replace('<span class="no">9</span>', '<span class="no">10</span>') \
    + '\n' + sec('timeouts').replace('<span class="no">10</span>', '<span class="no">11</span>') \
    + '\n' + sec('findings').replace('<span class="no">11</span>', '<span class="no">12</span>')

# script: keep tables, findings, sequence, theme; drop renderFlow usage
js = src[src.index('<script>'):]
js = re.sub(r"renderFlow\('f-[^']+',\[.*?\n\]\);\n", '', js, flags=re.S)
js = js.replace('</script>', '''
/* zoom controls */
document.querySelectorAll('.chartbox').forEach(box=>{
  const svg = box.querySelector('svg.flowsvg'), W = +svg.dataset.w, wrap = box.querySelector('.chartscroll');
  let z = 1;
  const apply = ()=>{ svg.style.width = Math.round(W*z)+'px'; svg.style.minWidth = '0'; };
  const fit = ()=>{ z = Math.max(0.7, Math.min(1, (wrap.clientWidth-4)/W)); apply(); };
  box.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
    const k=b.dataset.z; if(k==='in') z=Math.min(2.5,z*1.25); else if(k==='out') z=Math.max(0.3,z/1.25); else if(k==='one') z=1; else fit(); apply();
  }));
  fit();
});
addEventListener('resize',()=>{});
</script>''')
# the sequence svg has min-width style set in JS, keep
html = head + nav + '\n<main>\n' + body + '\n</main>\n' + js + '\n</body>\n</html>\n'
html = html.replace('<title>Plate Pipeline Map</title>', '<title>Plate Pipeline Map</title>')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline_map.html'), 'w').write(html)
print(len(html))
