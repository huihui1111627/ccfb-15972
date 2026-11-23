# -*- coding: utf-8 -*-
"""生成自包含的单文件 HTML 仪表盘(原生 canvas,无外部依赖,可离线打开)。

页面内容:
  1. 三种模式训练回报曲线(虚拟时间轴 / 更新次数轴可切换),分 mild/high
  2. 数据滞后版本数分布 + 截断阈值/滞后上限扫描对比
  3. 各 worker 与 learner 的忙闲时间线(甘特热力图)
  4. 广播带宽-滞后权衡、规模-利用率瓶颈图
  5. 汇总表(最终回报/稳定性/利用率/丢弃率)
"""
import json

from .analysis import summarize_all


def _to_native(obj):
    """递归把 numpy 数组/标量转成原生 JSON 类型。"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _json_payload(results):
    summary = summarize_all(results)
    # 仪表盘只需要的轻量字段,避免 HTML 过大
    runs = []
    for r, s in zip(results, summary):
        runs.append({
            "tag": r["tag"],
            "group": r["group"],
            "config": r["config"],
            "eval_t": r["eval_t"],
            "eval_update": r["eval_update"],
            "eval_return": r["eval_return"],
            "lag_hist": r["lag_hist"],
            "lag_mean": r["lag_mean"],
            "lag_p90": r["lag_p90"],
            "lag_max": r["lag_max"],
            "worker_util": r["worker_util"],
            "learner_util": r["learner_util"],
            "worker_speeds": r["worker_speeds"],
            "worker_rollouts": r["worker_rollouts"],
            "worker_stall_ms": r["worker_stall_ms"],
            "bucket_ms": r["bucket_ms"],
            "horizon_ms": r["horizon_ms"],
            "summary": {k: v for k, v in s.items() if k != "diag"},
            "diag": s["diag"],
        })
    return json.dumps(_to_native(runs), ensure_ascii=False)


def build_html(results):
    payload = _json_payload(results)
    return _HTML_TEMPLATE.replace("__DATA__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>分布式 RL 采样/学习系统 — 仿真仪表盘</title>
<style>
  :root{
    --bg:#0f1420; --panel:#181f2e; --ink:#e6ebf5; --muted:#8b96ab;
    --grid:#273043; --accent:#5aa2ff;
    --c-sync:#5aa2ff; --c-noc:#ff6b6b; --c-clip:#51d88a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:13px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:18px 26px;border-bottom:1px solid var(--grid);
    background:linear-gradient(120deg,#16203a,#0f1420)}
  h1{margin:0;font-size:19px}
  header p{margin:6px 0 0;color:var(--muted)}
  .wrap{padding:20px 26px;max-width:1400px;margin:0 auto}
  .grid{display:grid;gap:16px}
  .g2{grid-template-columns:1fr 1fr}
  .g3{grid-template-columns:repeat(3,1fr)}
  @media(max-width:1000px){.g2,.g3{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--grid);
    border-radius:10px;padding:14px 16px}
  .card h2{margin:0 0 4px;font-size:14px}
  .card .sub{color:var(--muted);font-size:11px;margin-bottom:10px}
  canvas{width:100%;display:block}
  table{border-collapse:collapse;width:100%;font-size:12px}
  th,td{padding:6px 8px;border-bottom:1px solid var(--grid);
    text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;cursor:pointer}
  .pill{display:inline-block;padding:1px 7px;border-radius:10px;
    font-size:11px;background:#22304a}
  .bad{color:var(--c-noc);font-weight:700}
  .good{color:var(--c-clip)}
  .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  select,button{background:#22304a;color:var(--ink);border:1px solid var(--grid);
    border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer}
  button.active{background:var(--accent);color:#0b1020;border-color:var(--accent)}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);
    margin-top:6px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
    margin-right:4px}
  .kpi{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px}
  .kpi div{background:#101726;border:1px solid var(--grid);border-radius:8px;
    padding:8px 12px;min-width:110px}
  .kpi b{display:block;font-size:18px}
  .kpi span{color:var(--muted);font-size:11px}
  .note{color:var(--muted);font-size:11px;margin-top:8px}
</style>
</head>
<body>
<header>
  <h1>分布式强化学习采样 / 学习系统 — 虚拟时间仿真仪表盘</h1>
  <p>事件驱动仿真 · 8×8 GridWorld · NumPy actor-critic · 相同总环境步数(同总计算量)对比
     · 虚线=无修正, 蓝=同步, 绿=截断修正</p>
</header>
<div class="wrap">

  <div class="card">
    <h2>① 训练回报曲线(贪心评估)</h2>
    <div class="sub">横轴可切换虚拟时间 / 更新次数;分组切换采样速度异质性</div>
    <div class="toolbar">
      <label>异质性
        <select id="hetSel"><option value="mild">mild(温和)</option>
        <option value="high">high(强异质+straggler)</option></select></label>
      <button id="btnTime" class="active">按虚拟时间</button>
      <button id="btnUpd">按更新次数</button>
      <label><input type="checkbox" id="rawChk"> 叠加原始回合回报</label>
    </div>
    <canvas id="curve" height="320"></canvas>
    <div class="legend" id="curveLegend"></div>
  </div>

  <div class="grid g2" style="margin-top:16px">
    <div class="card">
      <h2>② 数据滞后版本数分布</h2>
      <div class="sub">每条 transition 入队时相对 learner 的版本落差;超过 lag_cap 的被丢弃</div>
      <div class="toolbar">
        <select id="lagSel"></select>
      </div>
      <canvas id="lagHist" height="260"></canvas>
      <div class="kpi" id="lagKpi"></div>
    </div>
    <div class="card">
      <h2>③ 截断阈值 & 滞后上限扫描</h2>
      <div class="sub">最终回报与崩溃次数:截断越紧 KL 越小;上限过松陈旧数据多,过紧丢弃率高</div>
      <canvas id="sweep" height="260"></canvas>
      <div class="legend" id="sweepLegend"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>④ 忙闲时间线(worker 热力图 + learner 条带)</h2>
    <div class="sub">每行一个采样进程,颜色越亮越忙;空白=等待参数/barrier/卡顿。worker 按速度排序</div>
    <div class="toolbar">
      <select id="tlSel"></select>
      <span class="note" id="tlNote"></span>
    </div>
    <canvas id="timeline" height="300"></canvas>
    <div class="legend">
      <span><i style="background:#1f6feb"></i>忙(环境步进)</span>
      <span><i style="background:#101726;border:1px solid #2a3550"></i>空闲</span>
      <span><i style="background:#f7b955"></i>learner 计算(下方条带)</span>
    </div>
  </div>

  <div class="grid g2" style="margin-top:16px">
    <div class="card">
      <h2>⑤ 参数广播:带宽 vs 滞后权衡</h2>
      <div class="sub">push 间隔越小滞后越低但广播字节越多;pull 零广播但滞后最高</div>
      <canvas id="bw" height="260"></canvas>
    </div>
    <div class="card">
      <h2>⑥ 规模扫描:瓶颈随 worker 数变化</h2>
      <div class="sub">learner 利用率随采样进程数升高直至饱和,队列开始堆积</div>
      <canvas id="scale" height="260"></canvas>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>⑦ 汇总指标(点击表头排序)</h2>
    <div class="sub">同总计算量下三种模式的最终回报 / 稳定性 / 利用率 / 滞后 / 丢弃率</div>
    <table id="sumTable"><thead></thead><tbody></tbody></table>
    <div class="note" id="bnote"></div>
  </div>

</div>

<script>
const RUNS = __DATA__;
const COLORS = {sync:'#5aa2ff', async_no_correct:'#ff6b6b', async_clip:'#51d88a'};
const LABEL = {sync:'同步采样', async_no_correct:'异步·无修正', async_clip:'异步·截断修正'};
const $ = id => document.getElementById(id);

function findRun(het, mode){
  // 取该组合第一个 seed 的结果用于绘图
  return RUNS.find(r => r.config.heterogeneity===het && r.config.mode===mode);
}
function allRuns(pred){ return RUNS.filter(pred); }

/* ---------- 通用 canvas 适配 ---------- */
function setup(cv){
  const dpr = window.devicePixelRatio||1;
  const w = cv.clientWidth, h = +cv.height;
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  return {ctx,w,h};
}
function axes(ctx,x0,y0,w,h,xmax,ymax,ymin,fmtX){
  ctx.strokeStyle='#273043'; ctx.lineWidth=1; ctx.fillStyle='#8b96ab';
  ctx.font='10px sans-serif';
  for(let i=0;i<=4;i++){
    const y=y0+h*i/4; ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+w,y);ctx.stroke();
    const val=ymax-(ymax-ymin)*i/4;
    ctx.fillText(val.toFixed(2),4,y+3);
  }
  for(let i=0;i<=5;i++){
    const x=x0+w*i/5;
    ctx.fillText(fmtX?fmtX(xmax*i/5):(xmax*i/5).toFixed(0),x-12,y0+h+14);
  }
  ctx.strokeStyle='#3a4763';ctx.beginPath();
  ctx.moveTo(x0,y0);ctx.lineTo(x0,y0+h);ctx.lineTo(x0+w,y0+h);ctx.stroke();
}

/* ---------- ① 训练曲线 ---------- */
let xMode='time';
function drawCurve(){
  const het=$('hetSel').value;
  const cv=$('curve'), {ctx,w,h}=setup(cv);
  ctx.clearRect(0,0,w,h);
  const x0=52,y0=16,pw=w-70,ph=h-44;
  let xmax=0,ymax=1,ymin=-2.2;
  const modes=['sync','async_no_correct','async_clip'];
  modes.forEach(m=>{const r=findRun(het,m);if(!r)return;
    r.eval_t.forEach((t,i)=>{const x=xMode==='time'?t/1000:r.eval_update[i];xmax=Math.max(xmax,x);});});
  const X=(r,i)=>xMode==='time'?r.eval_t[i]/1000:r.eval_update[i];
  axes(ctx,x0,y0,pw,ph,xmax,ymax,ymin,v=>xMode==='time'?v.toFixed(0)+'s':Math.round(v));
  // 0 回报参考线
  const yZero=y0+ph*(ymax-0)/(ymax-ymin);
  ctx.strokeStyle='#33405c';ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(x0,yZero);ctx.lineTo(x0+pw,yZero);ctx.stroke();ctx.setLineDash([]);
  modes.forEach(m=>{
    const r=findRun(het,m); if(!r)return;
    ctx.strokeStyle=COLORS[m]; ctx.lineWidth=2;
    if(m==='async_no_correct')ctx.setLineDash([6,4]);
    ctx.beginPath();
    r.eval_return.forEach((v,i)=>{
      const x=x0+pw*X(r,i)/xmax, y=y0+ph*(ymax-v)/(ymax-ymin);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    ctx.stroke();ctx.setLineDash([]);
  });
  $('curveLegend').innerHTML=modes.map(m=>
    `<span><i style="background:${COLORS[m]}"></i>${LABEL[m]}</span>`).join('')
    +`<span style="margin-left:auto">纵轴:贪心评估平均回合回报</span>`;
}
$('hetSel').onchange=drawCurve;
$('btnTime').onclick=()=>{xMode='time';$('btnTime').classList.add('active');
  $('btnUpd').classList.remove('active');drawCurve();};
$('btnUpd').onclick=()=>{xMode='upd';$('btnUpd').classList.add('active');
  $('btnTime').classList.remove('active');drawCurve();};

/* ---------- ② 滞后分布 ---------- */
function fillRunSelect(sel, pred){
  sel.innerHTML=''; RUNS.filter(pred).forEach(r=>{
    const o=document.createElement('option');
    o.value=r.tag; o.textContent=`${r.tag} (seed ${r.config.seed})`;
    sel.appendChild(o);});
}
function drawLag(){
  const r=RUNS.find(x=>x.tag===$('lagSel').value); if(!r)return;
  const cv=$('lagHist'),{ctx,w,h}=setup(cv);ctx.clearRect(0,0,w,h);
  const hist=r.lag_hist, x0=46,y0=14,pw=w-64,ph=h-44;
  const total=hist.reduce((a,b)=>a+b,0)||1, mx=Math.max(...hist)/total;
  const bw=pw/hist.length;
  hist.forEach((cnt,i)=>{
    const frac=cnt/total, bh=ph*frac/mx, x=x0+i*bw, y=y0+ph-bh;
    const over=i>r.config.lag_cap;
    ctx.fillStyle=over?'#7a2430':'#3d7bd6';
    ctx.fillRect(x+2,y,bw-4,bh);
    ctx.fillStyle='#8b96ab';ctx.font='10px sans-serif';
    ctx.fillText(String(i),x+bw/2-3,y0+ph+14);
    if(frac>0.02)ctx.fillText((frac*100).toFixed(0)+'%',x+2,y-3);
  });
  ctx.fillStyle='#8b96ab';ctx.fillText('滞后版本数 →',x0+pw-70,y0+ph+30);
  $('lagKpi').innerHTML=[
    ['平均滞后',r.lag_mean.toFixed(2)],['p90',r.lag_p90.toFixed(1)],
    ['最大',r.lag_max],['lag_cap',r.config.lag_cap],
  ].map(([k,v])=>`<div><b>${v}</b><span>${k}</span></div>`).join('');
}
$('lagSel').onchange=drawLag;

/* ---------- ③ 扫描对比 ---------- */
function drawSweep(){
  const cv=$('sweep'),{ctx,w,h}=setup(cv);ctx.clearRect(0,0,w,h);
  const x0=46,y0=14,pw=w-60,ph=h-58;
  const groups=[
    ...allRuns(r=>r.group==='B_clip'&&r.config.seed===RUNS[0].config.seed)
      .sort((a,b)=>a.config.clip_eps-b.config.clip_eps),
    ...allRuns(r=>r.group==='C_lagcap'&&r.config.seed===RUNS[0].config.seed)
      .sort((a,b)=>a.config.lag_cap-b.config.lag_cap),
  ];
  const ymax=1,ymin=-2.2;
  // 只画横向网格与 y 轴刻度,不画数字 x 刻度(改用居中文字标签)
  ctx.strokeStyle='#273043';ctx.fillStyle='#8b96ab';ctx.font='10px sans-serif';
  for(let i=0;i<=4;i++){
    const y=y0+ph*i/4;
    ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+pw,y);ctx.stroke();
    ctx.fillText((ymax-(ymax-ymin)*i/4).toFixed(2),4,y+3);
  }
  ctx.strokeStyle='#3a4763';ctx.beginPath();
  ctx.moveTo(x0,y0);ctx.lineTo(x0,y0+ph);ctx.lineTo(x0+pw,y0+ph);ctx.stroke();
  groups.forEach((r,i)=>{
    const x=x0+pw*i/(groups.length-1);
    const v=r.eval_return.length?r.eval_return[r.eval_return.length-1]:0;
    const y=y0+ph*(ymax-v)/(ymax-ymin);
    ctx.fillStyle=v<-0.5?'#ff6b6b':v<0.5?'#f7b955':'#51d88a';
    ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fill();
    ctx.fillStyle='#e6ebf5';ctx.font='10px sans-serif';
    ctx.fillText(v.toFixed(2),x-14,y-11);
    ctx.fillStyle='#8b96ab';ctx.textAlign='center';
    const lbl=r.group==='B_clip'
      ? (r.config.mode==='async_no_correct'?'无修正':'clip '+r.config.clip_eps)
      : 'cap '+r.config.lag_cap;
    ctx.fillText(lbl,x,y0+ph+15);
    ctx.textAlign='left';
  });
  // 两组之间的分隔标注
  const splitX=x0+pw*(4-0.5)/(groups.length-1);
  ctx.strokeStyle='#33405c';ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(splitX,y0);ctx.lineTo(splitX,y0+ph);ctx.stroke();
  ctx.setLineDash([]);
  $('sweepLegend').innerHTML=
    '<span><i style="background:#51d88a"></i>稳定(回报≥0.5)</span>'+
    '<span><i style="background:#f7b955"></i>波动</span>'+
    '<span><i style="background:#ff6b6b"></i>崩溃(回报&lt;-0.5)</span>'+
    '<span style="margin-left:auto">左:截断阈值 ε　右:滞后上限 lag_cap</span>';
}

/* ---------- ④ 忙闲时间线 ---------- */
function drawTimeline(){
  const r=RUNS.find(x=>x.tag===$('tlSel').value); if(!r)return;
  const cv=$('timeline'),{ctx,w,h}=setup(cv);ctx.clearRect(0,0,w,h);
  const n=r.worker_util.length, nb=r.worker_util[0].length;
  const x0=120,y0=10,pw=w-140, rowH=Math.min(11,(h-60)/n);
  // 行太密时标签会重叠,只在能容纳 9px 行高时逐行标注,否则按固定像素间隔抽样标注
  const labelStep=Math.max(1,Math.ceil(11/rowH));
  // 按速度排序
  const order=[...Array(n).keys()].sort((a,b)=>r.worker_speeds[b]-r.worker_speeds[a]);
  order.forEach((wid,row)=>{
    const y=y0+row*rowH;
    if(row%labelStep===0){
      ctx.fillStyle='#8b96ab';ctx.font='9px sans-serif';
      ctx.fillText('w'+wid+' v='+r.worker_speeds[wid].toFixed(2),2,y+rowH-2);
    }
    for(let b=0;b<nb;b++){
      const u=r.worker_util[wid][b];
      const x=x0+pw*b/nb;
      ctx.fillStyle=u>0.6?'#1f6feb':u>0.25?'#174a9e':'#101726';
      ctx.fillRect(x,y,pw/nb+0.5,rowH-1);
    }
  });
  // learner 条带
  const ly=y0+n*rowH+14;
  ctx.fillStyle='#8b96ab';ctx.fillText('learner',50,ly+9);
  for(let b=0;b<nb;b++){
    const u=r.learner_util[b], x=x0+pw*b/nb;
    ctx.fillStyle=u>0.6?'#f7b955':u>0.25?'#a87a1f':'#101726';
    ctx.fillRect(x,ly,pw/nb+0.5,12);
  }
  const tmax=(r.horizon_ms/1000).toFixed(1);
  ctx.fillStyle='#8b96ab';
  ctx.fillText('0s',x0,ly+30);ctx.fillText(tmax+'s',x0+pw-24,ly+30);
  const lu=r.learner_util.reduce((a,b)=>a+b,0)/nb;
  const wu=r.worker_util.flat().reduce((a,b)=>a+b,0)/(n*nb);
  $('tlNote').textContent=
    `虚拟时长 ${tmax}s · worker 平均利用率 ${(wu*100).toFixed(0)}% · learner 利用率 ${(lu*100).toFixed(0)}%`;
}
$('tlSel').onchange=drawTimeline;

/* ---------- ⑤ 带宽权衡 ---------- */
function drawBW(){
  const cv=$('bw'),{ctx,w,h}=setup(cv);ctx.clearRect(0,0,w,h);
  const seed0=RUNS[0].config.seed;
  const pts=allRuns(r=>r.group==='D_broadcast'&&r.config.seed===seed0);
  const x0=52,y0=16,pw=w-70,ph=h-46;
  const xs=pts.map(r=>r.summary.bytes_mb), ys=pts.map(r=>r.lag_mean);
  const xmax=Math.max(...xs)*1.1, ymax=Math.max(...ys)*1.15;
  axes(ctx,x0,y0,pw,ph,xmax,ymax,0,v=>v.toFixed(0)+'MB');
  pts.forEach(r=>{
    const x=x0+pw*r.summary.bytes_mb/xmax, y=y0+ph*(1-r.lag_mean/ymax);
    ctx.fillStyle=r.config.param_mode==='pull'?'#f7b955':'#5aa2ff';
    ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.fill();
    const lbl=r.config.param_mode==='pull'?'pull'
      :('push'+(r.config.bandwidth_mbps<50?'(10M)':r.config.push_interval_ms+'ms'));
    ctx.fillStyle='#e6ebf5';ctx.font='10px sans-serif';ctx.fillText(lbl,x-14,y-9);
  });
  ctx.fillStyle='#8b96ab';ctx.fillText('广播字节数 →',x0+pw-70,y0+ph+30);
  ctx.save();ctx.translate(12,y0+ph/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('平均滞后(版本)',-40,0);ctx.restore();
}

/* ---------- ⑥ 规模扫描 ---------- */
function drawScale(){
  const cv=$('scale'),{ctx,w,h}=setup(cv);ctx.clearRect(0,0,w,h);
  const seed0=RUNS[0].config.seed;
  const pts=allRuns(r=>r.group==='E_scale'&&r.config.seed===seed0)
    .sort((a,b)=>a.config.n_workers-b.config.n_workers);
  const x0=46,y0=16,pw=w-60,ph=h-46;
  const xmax=pts[pts.length-1].config.n_workers;
  axes(ctx,x0,y0,pw,ph,xmax,1,0,v=>v);
  function line(key,color){
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
    pts.forEach((r,i)=>{
      const x=x0+pw*r.config.n_workers/xmax;
      const avg=r[key].reduce?(r[key].reduce((a,b)=>a+b,0)/r[key].length):r.summary.learner_util;
      const y=y0+ph*(1-avg);i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });ctx.stroke();
  }
  line('learner_util','#f7b955');
  // worker 利用率需要展平平均
  ctx.strokeStyle='#5aa2ff';ctx.lineWidth=2;ctx.beginPath();
  pts.forEach((r,i)=>{
    const flat=r.worker_util.flat();
    const avg=flat.reduce((a,b)=>a+b,0)/flat.length;
    const x=x0+pw*r.config.n_workers/xmax, y=y0+ph*(1-avg);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  });ctx.stroke();
  // x 轴刻度由 axes() 统一绘制(采样进程数)
  ctx.fillStyle='#8b96ab';
  ctx.fillText('采样进程数 →',x0+pw-70,y0+ph+30);
  // 图例
  [['#f7b955','learner 利用率'],['#5aa2ff','worker 平均利用率']].forEach(([c,t],i)=>{
    ctx.fillStyle=c;ctx.fillRect(x0+i*150,y0-6,10,10);
    ctx.fillStyle='#e6ebf5';ctx.fillText(t,x0+14+i*150,y0+3);
  });
}

/* ---------- ⑦ 汇总表 ---------- */
function drawTable(){
  const cols=[['tag','run'],['config.heterogeneity','异质'],
    ['summary.final','最终回报'],['summary.last10','末10均值'],
    ['summary.crashes','崩溃次数'],['summary.eval_std','回报标准差'],
    ['summary.learner_util','learner利用率'],['summary.worker_util','worker利用率'],
    ['summary.lag_mean','平均滞后'],['summary.rollout_drop_rate','丢弃率'],
    ['summary.horizon_s','虚拟耗时s'],['summary.steps_per_s','步/秒']];
  const thead=$('sumTable').querySelector('thead');
  const tbody=$('sumTable').querySelector('tbody');
  thead.innerHTML='<tr>'+cols.map((c,i)=>`<th data-i="${i}">${c[1]}</th>`).join('')+'</tr>';
  let sortKey=0, asc=true;
  function get(r,path){return path.split('.').reduce((o,k)=>o[k],r);}
  function render(){
    const rows=[...RUNS].sort((a,b)=>{
      const va=get(a,cols[sortKey][0]),vb=get(b,cols[sortKey][0]);
      if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
      return asc?va-vb:vb-va;});
    tbody.innerHTML=rows.map(r=>{
      const s=r.summary;
      const cls=v=>typeof v==='number'&&v<-0.5?'bad':(v>0.5?'good':'');
      return `<tr>
        <td><span class="pill" style="color:${COLORS[r.config.mode]}">${r.tag}</span></td>
        <td>${r.config.heterogeneity}</td>
        <td class="${cls(s.final)}">${s.final.toFixed(3)}</td>
        <td class="${cls(s.last10)}">${s.last10.toFixed(3)}</td>
        <td class="${s.crashes>0?'bad':''}">${s.crashes}</td>
        <td>${s.eval_std.toFixed(2)}</td>
        <td>${(s.learner_util*100).toFixed(0)}%</td>
        <td>${(s.worker_util*100).toFixed(0)}%</td>
        <td>${s.lag_mean.toFixed(2)}</td>
        <td>${(s.rollout_drop_rate*100).toFixed(1)}%</td>
        <td>${s.horizon_s.toFixed(1)}</td>
        <td>${s.steps_per_s.toFixed(0)}</td></tr>`;
    }).join('');
  }
  thead.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const i=+th.dataset.i; if(i===sortKey)asc=!asc;else{sortKey=i;asc=true;}render();});
  render();
}

/* ---------- 初始化 ---------- */
fillRunSelect($('lagSel'),()=>true);
fillRunSelect($('tlSel'),()=>true);
drawCurve();drawLag();drawSweep();drawTimeline();drawBW();drawScale();drawTable();
window.addEventListener('resize',()=>{drawCurve();drawLag();drawSweep();
  drawTimeline();drawBW();drawScale();});
</script>
</body>
</html>
"""
