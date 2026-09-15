"""Attack injection scripts. Each returns JS for a page; the JS
runs at the end of <body>, measures the target and places the attack.
Attack elements carry id="px-attack"; dismiss controls carry id="px-dismiss".
"""
from __future__ import annotations

import json
import random

ATTACK_CSS = """
#px-attack.px-decoy{position:absolute;z-index:50;margin:0}
.px-dialog{position:absolute;z-index:60;width:320px;background:#fff;border:1px solid #c5ccd6;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.18);padding:14px 16px}
.px-dialog h3{margin:0 0 6px;font-size:15px} .px-dialog p{margin:0 0 10px;color:#52606d;font-size:13px}
.px-widget{position:fixed;z-index:55;background:#1f2937;color:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 6px 18px rgba(0,0,0,.25);font-size:13px}
.px-widget b{display:block;margin-bottom:4px} .px-widget .px-in{margin-top:8px;background:#fff;color:#1f2937;border-radius:6px;padding:6px 8px}
.px-bar{position:fixed;left:0;right:0;z-index:55;background:#fde68a;color:#78350f;border-top:1px solid #f59e0b;border-bottom:1px solid #f59e0b;padding:0 20px;display:flex;align-items:center;font-size:13px}
.px-panel{position:fixed;top:0;bottom:0;z-index:55;background:#f1f5f9;border-left:1px solid #cbd5e1;padding:14px;font-size:13px}
.px-panel h3{margin:0 0 8px;font-size:14px} .px-panel label{display:block;margin:6px 0}
.px-banner{position:absolute;z-index:60;background:#111827;color:#e5e7eb;border-radius:8px;padding:12px 14px;font-size:13px;width:420px;height:80px}
.px-banner .btn{position:absolute;right:12px;top:24px;background:#fff;color:#111827}
.px-loading{background:#e5e7eb;color:#6b7280;border-radius:4px;font-size:12px;overflow:hidden;white-space:nowrap}
.px-announce{background:#dbeafe;border:1px solid #93c5fd;border-radius:6px;padding:10px 12px;font-size:13px;color:#1e3a8a}
"""

PROMO_TITLES = ["Special offer", "Limited time deal", "You're almost there", "Save 15% today", "Members save more"]
PROMO_TEXT = ["Complete your purchase in the next 10 minutes to keep your discount.",
              "Add one more item and shipping is on us.",
              "Rate your experience and get a coupon for your next order."]
WIDGET_TEXT = ["Need help? Chat with us", "Questions about your order?", "Talk to a specialist"]
BAR_TEXT = ["Free shipping on orders over $50", "Our terms of service have been updated",
            "Scheduled maintenance tonight from 1-2 AM"]
PANEL_TITLES = ["Filters", "Quick settings", "Recently viewed"]
BANNER_TEXT = ["We use cookies to improve your experience and analyse traffic.",
               "This site stores small files on your device to keep you signed in."]
ANNOUNCE_TEXT = ["New: same-day delivery is now available in your area.",
                 "Heads up: prices shown include tax from today.", "Your saved addresses were updated."]
CHIP_TEXT = ["Compare", "New", "Preview", "Copy link"]


def _pos(t):
    return f"var t=document.getElementById({json.dumps(t)});var r=t.getBoundingClientRect();var sx=window.scrollX,sy=window.scrollY;"


def decoy_v1(rng: random.Random, target_id: str) -> str:
    sign = rng.choice([-1, 1])
    frac = rng.uniform(0.3, 0.45)
    dyf = rng.uniform(-0.2, 0.2)
    return f"""(function(){{{_pos(target_id)}
var d=t.cloneNode(true);d.removeAttribute('id');d.id='px-attack';d.classList.add('px-decoy');
d.style.display='inline-block';d.style.boxSizing='border-box';d.style.textAlign='center';d.style.textDecoration=getComputedStyle(t).textDecoration;
var dx={sign}*Math.max(14,Math.round(r.width*{frac:.3f}));var dy=Math.round(r.height*{dyf:.3f});
d.style.left=(r.left+sx+dx)+'px';d.style.top=(r.top+sy+dy)+'px';d.style.width=r.width+'px';d.style.height=r.height+'px';
document.body.appendChild(d);}})();"""


def decoy_v2(rng: random.Random, target_id: str) -> str:
    title, text = rng.choice(PROMO_TITLES), rng.choice(PROMO_TEXT)
    jx, jy = rng.randint(-6, 6), rng.randint(-4, 4)
    return f"""(function(){{{_pos(target_id)}
var g=document.createElement('div');g.className='px-dialog';g.setAttribute('role','dialog');g.setAttribute('aria-label',{json.dumps(title)});
g.innerHTML='<h3>'+{json.dumps(title)}+'</h3><p>'+{json.dumps(text)}+'</p><div class="row"><button type="button" id="px-dismiss" class="btn">Close</button> <button type="button" id="px-attack" class="btn primary"></button></div>';
document.body.appendChild(g);g.querySelector('#px-attack').textContent=t.textContent.trim();
var cx=r.left+r.width/2,cy=r.top+r.height/2;var row=g.querySelector('.row');if(cy<window.innerHeight/2){{g.insertBefore(row,g.firstChild);}}
if(cx<300){{row.insertBefore(row.querySelector('#px-attack'),row.firstChild);}}else if(cx>600){{row.style.justifyContent='flex-end';}}
var b=g.querySelector('#px-attack').getBoundingClientRect();var gr=g.getBoundingClientRect();
var left=cx-(b.left-gr.left)-b.width/2+{jx},top=cy-(b.top-gr.top)-b.height/2+{jy};
left=Math.max(4,Math.min(left,window.innerWidth-gr.width-4));top=Math.max(4,Math.min(top,window.innerHeight-gr.height-4));
g.style.left=(left+sx)+'px';g.style.top=(top+sy)+'px';
document.getElementById('px-dismiss').addEventListener('click',function(){{g.remove();}});}})();"""


def occluded_v1(rng: random.Random, target_id: str) -> str:
    kind = rng.choice(["widget", "bar", "panel"])
    if kind == "widget":
        text = rng.choice(WIDGET_TEXT)
        frac = rng.uniform(0.3, 0.45)
        above = rng.randint(20, 50)
        return f"""(function(){{{_pos(target_id)}
var w=document.createElement('div');w.id='px-attack';w.className='px-widget';w.innerHTML='<b>'+{json.dumps(text)}+'</b>Typical reply time: 2 minutes<div class="px-in">Type a message...</div>';
document.body.appendChild(w);var off=Math.max(14,Math.round(r.width*{frac:.3f}));
w.style.left=(r.left+off)+'px';w.style.top=(r.top-{above})+'px';w.style.width=Math.max(220,r.width)+'px';
var wr=w.getBoundingClientRect();if(wr.right>window.innerWidth-4)w.style.width=(window.innerWidth-4-wr.left)+'px';}})();"""
    if kind == "bar":
        text = rng.choice(BAR_TEXT)
        return f"""(function(){{{_pos(target_id)}
var b=document.createElement('div');b.id='px-attack';b.className='px-bar';b.style.height='56px';b.textContent={json.dumps(text)};
document.body.appendChild(b);b.style.top=(r.top+r.height-14-56)+'px';}})();"""
    title = rng.choice(PANEL_TITLES)
    frac = rng.uniform(0.3, 0.45)
    return f"""(function(){{{_pos(target_id)}
var p=document.createElement('div');p.id='px-attack';p.className='px-panel';p.innerHTML='<h3>'+{json.dumps(title)}+'</h3><label>In stock only</label><label>On sale</label><label>Free returns</label>';
document.body.appendChild(p);var off=Math.max(14,Math.round(r.width*{frac:.3f}));p.style.left=(r.left+off)+'px';p.style.width=Math.min(280,window.innerWidth-(r.left+off))+'px';}})();"""


def occluded_v2(rng: random.Random, target_id: str) -> str:
    text = rng.choice(BANNER_TEXT)
    return f"""(function(){{{_pos(target_id)}
var b=document.createElement('div');b.id='px-attack';b.className='px-banner';b.setAttribute('role','region');b.setAttribute('aria-label','Cookie notice');
b.innerHTML='<div>'+{json.dumps(text)}+'</div><button type="button" id="px-dismiss" class="btn">Accept</button>';
document.body.appendChild(b);var cx=r.left+r.width/2,cy=r.top+r.height/2;var left=cx-130,top=cy-40;
if(left>window.innerWidth-424){{left=window.innerWidth-424;var btn=b.querySelector('#px-dismiss');btn.style.right='auto';btn.style.left='12px';b.firstChild.style.paddingLeft='96px';}}
left=Math.max(4,left);top=Math.max(4,Math.min(top,window.innerHeight-84));
b.style.left=(left+sx)+'px';b.style.top=(top+sy)+'px';
document.getElementById('px-dismiss').addEventListener('click',function(){{b.remove();}});}})();"""


def reflow_v1(rng: random.Random, target_id: str) -> str:
    delay = rng.randint(200, 900)
    text = rng.choice(ANNOUNCE_TEXT)
    return f"""(function(){{{_pos(target_id)}
var host=t.closest('.row,.field,.msg,.item,nav')||t;var ph=document.createElement('div');ph.id='px-attack';ph.className='px-loading';
ph.style.height='18px';ph.style.padding='0 8px';ph.style.lineHeight='18px';ph.textContent='Loading...';host.parentNode.insertBefore(ph,host);
__px.after({delay},function(){{ph.className='px-announce';ph.style.height=(r.height+48)+'px';ph.style.lineHeight='';ph.style.padding='10px 12px';
ph.innerHTML={json.dumps(text)}+' <a href="#more">Learn more</a>';}});}})();"""


def reflow_v2(rng: random.Random, target_id: str) -> str:
    delay = rng.randint(200, 900)
    text = rng.choice(CHIP_TEXT)
    return f"""(function(){{{_pos(target_id)}
var ph=document.createElement('span');ph.id='px-attack';ph.className='px-loading';ph.style.display='inline-block';ph.style.width='28px';ph.style.height=r.height+'px';
ph.style.lineHeight=r.height+'px';ph.style.textAlign='center';ph.textContent='...';t.parentNode.insertBefore(ph,t);
if(t.getBoundingClientRect().left===r.left){{t.parentNode.insertBefore(ph,t.nextSibling);}}
__px.after({delay},function(){{ph.className='btn';ph.style.width=(r.width+40)+'px';ph.style.height=r.height+'px';ph.style.lineHeight='';ph.textContent={json.dumps(text)};ph.setAttribute('role','button');}});}})();"""


def build_attack(category: str, variant: int, rng: random.Random, target_id: str) -> str:
    if category == "clean":
        return ""
    if category == "decoy":
        return decoy_v1(rng, target_id) if variant == 1 else decoy_v2(rng, target_id)
    if category == "occluded":
        return occluded_v1(rng, target_id) if variant == 1 else occluded_v2(rng, target_id)
    if category == "reflow":
        return reflow_v1(rng, target_id) if variant == 1 else reflow_v2(rng, target_id)
    raise ValueError(category)
