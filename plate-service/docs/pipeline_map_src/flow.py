"""tiny flowchart-to-SVG generator (orthogonal arrows, diamonds, queues, swimlanes)"""
import html

CW = 6.9     # px per char at 12.5px
LH = 15.5


def esc(t):
    return html.escape(t, quote=False)


KIND_CLASS = {
    'start': 'c-green', 'proc': 'c-blue', 'dec': 'c-amber', 'store': 'c-purple',
    'end': 'c-red', 'ext': 'c-grey', 'good': 'c-green', 'hub': 'c-amber',
}
EDGE_STYLE = {'n': 'e-n', 'y': 'e-y', 'x': 'e-x', 'd': 'e-d', 'l': 'e-l'}


class Chart:
    _count = 0

    def __init__(s, w, h, lanes=None, title=None):
        Chart._count += 1
        s.id = f"c{Chart._count}"
        s.w, s.h = w, h
        s.nodes, s.parts, s.edge_svg, s.node_svg = {}, [], [], []
        s.lanes = lanes or []
        s.title = title

    # ------------------------------------------------------------ nodes
    def n(s, i, x, y, text, kind='proc', minw=0, minh=0):
        lines = text.split('\n')
        bold = kind in ('start', 'end', 'store')
        tw = max(len(l) * (CW * 1.13 if (j == 0 and bold) else CW) for j, l in enumerate(lines))
        if kind == 'dec':
            w = max(minw, tw * 1.5 + 30)
            h = max(minh, 62, len(lines) * LH * 1.55 + 22)
        elif kind in ('start', 'end'):
            w = max(minw, tw + 40)
            h = max(minh, len(lines) * LH + 18)
        elif kind == 'store':
            w = max(minw, tw + 30)
            h = max(minh, len(lines) * LH + 30)
        else:
            w = max(minw, tw + 28)
            h = max(minh, len(lines) * LH + 16)
        s.nodes[i] = dict(x=x, y=y, w=w, h=h, kind=kind, lines=lines)
        return i

    def stack(s, x, y0, items, gap=34, minw=0):
        """items: (id, kind, text); returns ids. y0 = top of first node."""
        ids, top = [], y0
        for it in items:
            i, kind, text = it
            tmp = Chart.__new__(Chart)
            lines = text.split('\n')
            # compute height the same way as n()
            s.n(i, x, 0, text, kind, minw)
            h = s.nodes[i]['h']
            s.nodes[i]['y'] = top + h / 2
            top += h + gap
            ids.append(i)
        return ids

    def anchor(s, i, a):
        n = s.nodes[i]
        return {'t': (n['x'], n['y'] - n['h'] / 2), 'b': (n['x'], n['y'] + n['h'] / 2),
                'l': (n['x'] - n['w'] / 2, n['y']), 'r': (n['x'] + n['w'] / 2, n['y'])}[a]

    # ------------------------------------------------------------ edges
    def e(s, a, b=None, label='', sa=None, ta=None, via=None, st='n', pos=None, stub=None):
        na = s.nodes[a]
        if b is None:                       # stub: short arrow to nowhere
            sa = sa or 'r'
            p0 = s.anchor(a, sa)
            dx, dy = {'r': (1, 0), 'l': (-1, 0), 'b': (0, 1), 't': (0, -1)}[sa]
            ln = stub or 46
            pts = [p0, (p0[0] + dx * ln, p0[1] + dy * ln)]
        else:
            nb = s.nodes[b]
            if sa is None or ta is None:
                dx, dy = nb['x'] - na['x'], nb['y'] - na['y']
                if via is None and abs(dx) < 2:
                    sa, ta = ('b', 't') if dy > 0 else ('t', 'b')
                elif via is None and abs(dy) > abs(dx) * 1.2:
                    sa, ta = ('b', 't') if dy > 0 else ('t', 'b')
                else:
                    sa, ta = ('r', 'l') if dx > 0 else ('l', 'r')
            p0, pn = s.anchor(a, sa), s.anchor(b, ta)
            if via is not None:
                pts = [p0] + list(via) + [pn]
            else:
                pts = s._auto(p0, pn, sa, ta)
        if b is None and pos is None:
            pos = (stub or 46) * 0.5
        s._draw(pts, label, st, pos)

    @staticmethod
    def _auto(p0, pn, sa, ta):
        if abs(p0[0] - pn[0]) < 1.5 and sa in 'bt' and ta in 'bt':
            return [p0, pn]
        if abs(p0[1] - pn[1]) < 1.5 and sa in 'lr' and ta in 'lr':
            return [p0, pn]
        if sa in 'bt' and ta in 'bt':
            my = (p0[1] + pn[1]) / 2
            return [p0, (p0[0], my), (pn[0], my), pn]
        if sa in 'lr' and ta in 'lr':
            mx = (p0[0] + pn[0]) / 2
            return [p0, (mx, p0[1]), (mx, pn[1]), pn]
        if sa in 'lr' and ta in 'bt':
            return [p0, (pn[0], p0[1]), pn]
        return [p0, (p0[0], pn[1]), pn]

    def _draw(s, pts, label, st, pos):
        cls = EDGE_STYLE[st]
        d = 'M' + ' L'.join(f"{x:.1f} {y:.1f}" for x, y in pts)
        s.edge_svg.append(f'<path class="ed {cls}" d="{d}" marker-end="url(#{s.id}{st})"/>')
        if not label:
            return
        # label position: pos px from start (default 22 for yes/no, mid for others)
        segs, total = [], 0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            L = abs(x1 - x0) + abs(y1 - y0)
            segs.append((x0, y0, x1, y1, L))
            total += L
        if pos == 'end':
            pos = total - 34
        want = (pos if pos is not None else (min(24, total * 0.5) if st in ('y', 'x') else total / 2))
        want = min(want, total)
        acc = 0
        for x0, y0, x1, y1, L in segs:
            if acc + L >= want or (x0, y0, x1, y1, L) == segs[-1]:
                t = (want - acc) / L if L else 0
                px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                vertical = abs(x1 - x0) < abs(y1 - y0)
                break
            acc += L
        lines = label.split('\n')
        tcls = 'lb-' + st
        if vertical:
            for k, ln in enumerate(lines):
                s.edge_svg.append(f'<text class="lb {tcls}" x="{px + 7:.1f}" y="{py + 4 + k * 12:.1f}">{esc(ln)}</text>')
        else:
            for k, ln in enumerate(lines):
                s.edge_svg.append(f'<text class="lb {tcls}" x="{px:.1f}" y="{py - 6 - (len(lines) - 1 - k) * 12:.1f}" text-anchor="middle">{esc(ln)}</text>')

    # ------------------------------------------------------------ render
    def _node(s, i, n):
        x, y, w, h, k = n['x'], n['y'], n['w'], n['h'], n['kind']
        cls = KIND_CLASS[k]
        out = ''
        if k == 'dec':
            out += f'<polygon class="nd {cls}" points="{x},{y - h / 2} {x + w / 2},{y} {x},{y + h / 2} {x - w / 2},{y}"/>'
        elif k in ('start', 'end'):
            out += f'<rect class="nd {cls}" x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="{h / 2}"/>'
        elif k == 'store':
            ry = 7
            x0, y0 = x - w / 2, y - h / 2
            out += (f'<path class="nd {cls}" d="M{x0} {y0 + ry} A{w / 2} {ry} 0 0 1 {x0 + w} {y0 + ry} V{y0 + h - ry} '
                    f'A{w / 2} {ry} 0 0 1 {x0} {y0 + h - ry} Z"/>'
                    f'<path class="nd {cls}" style="fill:none" d="M{x0} {y0 + ry} A{w / 2} {ry} 0 0 0 {x0 + w} {y0 + ry}"/>')
        else:
            out += f'<rect class="nd {cls}" x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="7"/>'
        n_l = len(n['lines'])
        top = y - (n_l - 1) * LH / 2 + (4 if k != 'store' else 8)
        for j, ln in enumerate(n['lines']):
            cl = 'tt b' if (j == 0 and k in ('start', 'end', 'store')) else 'tt'
            out += f'<text class="{cl}" x="{x}" y="{top + j * LH:.1f}" text-anchor="middle">{esc(ln)}</text>'
        return out

    def fit(s):
        s.h = max(n['y'] + n['h'] / 2 for n in s.nodes.values()) + 40
        return s

    def svg(s):
        o = [f'<svg class="flowsvg" data-w="{s.w}" viewBox="0 0 {s.w} {s.h}" xmlns="http://www.w3.org/2000/svg" role="img">']
        o.append('<defs>' + ''.join(
            f'<marker id="{s.id}{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0L10 5L0 10z" class="mk-{k}"/></marker>' for k in EDGE_STYLE) + '</defs>')
        if s.lanes:
            edges = []
            for k, (name, cx, wd) in enumerate(s.lanes):
                o.append(f'<rect class="lane-band {"alt" if k % 2 else ""}" x="{cx - wd / 2}" y="0" width="{wd}" height="{s.h}"/>')
                o.append(f'<text class="lane-name" x="{cx}" y="24" text-anchor="middle">{esc(name)}</text>')
        if s.title:
            o.append(f'<text class="chart-title" x="14" y="22">{esc(s.title)}</text>')
        o.extend(s.edge_svg)
        for i, n in s.nodes.items():
            o.append(s._node(i, n))
        o.append('</svg>')
        return '\n'.join(o)
