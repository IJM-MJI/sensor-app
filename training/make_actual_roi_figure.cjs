const { chromium } = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');

const project = 'C:/Users/Administrator/Documents/ChatGPT/New project/sensor-app';
const opencv = 'C:/Users/Administrator/Documents/ChatGPT/New project/.tmp/deployed-sensor/opencv.js';
const source = 'D:/app_test/꼬리치레';
const paperOut = path.join(project, 'paper-figures', 'actual-roi-extraction');
const vizOut = 'C:/Users/Administrator/.codex/visualizations/2026/09/09/01a083f0-8486-71b0-ae4f-215ab9796481';
fs.mkdirSync(paperOut, { recursive: true });
fs.mkdirSync(vizOut, { recursive: true });

function esc(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--disable-background-networking', '--host-resolver-rules=MAP * ~NOTFOUND']
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1500, height: 1100 }, deviceScaleFactor: 1 });
    page.setDefaultTimeout(60000);
    await page.route('**/*', route => {
      const req = route.request();
      const u = new URL(req.url());
      if (u.href === 'https://docs.opencv.org/4.9.0/opencv.js') {
        return route.fulfill({ path: opencv, contentType: 'application/javascript' });
      }
      if (u.origin !== 'https://local.sensor.test' || req.method() !== 'GET') return route.abort();
      const rel = u.pathname.slice(1) || 'index.html';
      if (!/^(index\.html|sensor-[\w-]+\.js)$/.test(rel)) return route.abort();
      const file = path.join(project, rel);
      return fs.existsSync(file)
        ? route.fulfill({ path: file, contentType: rel.endsWith('.html') ? 'text/html' : 'application/javascript' })
        : route.fulfill({ status: 404, body: '' });
    });
    await page.goto('https://local.sensor.test/', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => typeof cv !== 'undefined' && cv.Mat && typeof processFrame === 'function' && document.getElementById('loadingScreen').classList.contains('hidden'));

    await page.evaluate(() => {
      window.__figureResult = null;
      const original = processFrame;
      processFrame = function () {
        const result = original.apply(this, arguments);
        window.__figureResult = result;
        return result;
      };
    });

    await page.locator('#testImageInput').setInputFiles(path.join(source, 'calibration.png'));
    await page.waitForFunction(() => calibrated && !calMode && !captured);
    await page.evaluate(() => { if (captured) retake(); window.__figureResult = null; });
    await page.locator('#testImageInput').setInputFiles(path.join(source, '005.png'));
    await page.waitForFunction(() => window.__figureResult && document.getElementById('procBEL') === null && document.getElementById('procOverlay').classList.contains('hidden'));

    const figureData = await page.evaluate(() => {
      const result = window.__figureResult;
      const g = result._dbgGeo;
      const pixels = g.pixels;
      const W = g.w, H = g.h;
      const base = document.createElement('canvas'); base.width = W; base.height = H;
      base.getContext('2d').drawImage(cvs, 0, 0, W, H);
      function canvas() { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; }
      function opaqueBlack(imageData) { for (let i=3;i<imageData.data.length;i+=4) imageData.data[i]=255; return imageData; }
      function encode(c, quality = .86) {
        const o = document.createElement('canvas'); o.width = 330; o.height = 330;
        const x = o.getContext('2d'); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high';
        x.drawImage(c, 0, 0, W, H, 0, 0, 330, 330);
        return o.toDataURL('image/webp', quality);
      }
      function cmap(t) {
        t = Math.max(0, Math.min(1, t));
        const stops = [[0, 8, 5, 35], [.25, 58, 12, 110], [.5, 150, 32, 105], [.75, 238, 105, 45], [1, 252, 245, 165]];
        let j = 0; while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
        const a = stops[j], b = stops[j + 1], q = (t - a[0]) / (b[0] - a[0]);
        return [Math.round(a[1] + q * (b[1] - a[1])), Math.round(a[2] + q * (b[2] - a[2])), Math.round(a[3] + q * (b[3] - a[3]))];
      }
      function quantile(values, q) { const a = values.slice().sort((x, y) => x - y); return a[Math.max(0, Math.min(a.length - 1, Math.floor((a.length - 1) * q)))]; }

      const original = encode(base, .90);

      const roi = canvas(); const rx = roi.getContext('2d'); rx.drawImage(base, 0, 0);
      rx.lineWidth = 3; rx.strokeStyle = '#ef4444'; rx.beginPath(); rx.arc(g.cx, g.cy, g.r, 0, Math.PI * 2); rx.stroke();
      rx.lineWidth = 2; rx.strokeStyle = '#facc15'; rx.beginPath(); rx.arc(g.cx, g.cy, g.innerR, 0, Math.PI * 2); rx.stroke();
      rx.setLineDash([6, 4]); rx.strokeStyle = '#fb923c'; rx.strokeRect(g.cx - .55*g.r, g.cy - .62*g.r, .90*g.r, .76*g.r);
      rx.strokeStyle = '#22d3ee'; rx.strokeRect(g.cx - .55*g.r, g.cy + .18*g.r, .90*g.r, .50*g.r); rx.setLineDash([]);

      const cs = pixels.map(p => p.c), cLo = quantile(cs, .05), cHi = quantile(cs, .99);
      const chroma = canvas(), cd = opaqueBlack(chroma.getContext('2d').createImageData(W, H));
      for (const p of pixels) { const z = cmap((p.c - cLo) / Math.max(cHi - cLo, 1e-6)); const i = (p.y * W + p.x) * 4; cd.data[i]=z[0];cd.data[i+1]=z[1];cd.data[i+2]=z[2];cd.data[i+3]=255; }
      chroma.getContext('2d').putImageData(cd, 0, 0);

      const nBg = Math.floor(pixels.length * .5), bg = pixels.slice(0, nBg);
      const bgL = bg.reduce((s,p)=>s+p.L,0)/bg.length, bgA = bg.reduce((s,p)=>s+p.a,0)/bg.length, bgB = bg.reduce((s,p)=>s+p.b,0)/bg.length;
      function dist(p){ const dl=.35*(p.L-bgL), da=p.a-bgA, db=p.b-bgB; return Math.sqrt(dl*dl+da*da+db*db); }
      const inFlame = p => p.nx>=-.55&&p.nx<=.35&&p.ny>=-.62&&p.ny<=.14;
      const reg = result._raw._dropRegistration;
      const inDrop = p => {
        if (!reg) return p.nx>=-.55&&p.nx<=.35&&p.ny>=.18&&p.ny<=.68;
        const dx=p.nx-reg.x,dy=p.ny-reg.y,co=Math.cos(reg.angle),si=Math.sin(reg.angle);
        const lx=co*dx-si*dy,ly=si*dx+co*dy;
        return (lx/.25)**2+(ly/.29)**2<=1 || ((lx-.30)/.14)**2+((ly-.02)/.18)**2<=1;
      };
      function select(zone) {
        const a = pixels.filter(zone).map(p=>({p,d:dist(p)})).sort((x,y)=>x.d-y.d);
        let out = a.slice(Math.floor(a.length*.65)).filter(x=>x.d>=6).map(x=>x.p);
        if(out.length<12) out=a.slice(Math.max(0,a.length-Math.max(12,Math.floor(a.length*.2)))).map(x=>x.p);
        return out;
      }
      const flame = select(inFlame), drop = select(inDrop);
      const zoneScores = pixels.filter(p=>inFlame(p)||inDrop(p)).map(dist), dLo=quantile(zoneScores,.05), dHi=quantile(zoneScores,.99);
      const score = canvas(), sd=opaqueBlack(score.getContext('2d').createImageData(W,H));
      for(const p of pixels){ if(!inFlame(p)&&!inDrop(p))continue; const z=cmap((dist(p)-dLo)/Math.max(dHi-dLo,1e-6)); const i=(p.y*W+p.x)*4; sd.data[i]=z[0];sd.data[i+1]=z[1];sd.data[i+2]=z[2];sd.data[i+3]=255; }
      score.getContext('2d').putImageData(sd,0,0);

      const extracted=canvas(), ex=extracted.getContext('2d'), ed=opaqueBlack(ex.createImageData(W,H));
      for(const p of bg){const i=(p.y*W+p.x)*4; ed.data[i]=150;ed.data[i+1]=150;ed.data[i+2]=150;ed.data[i+3]=125;}
      for(const p of pixels.slice(Math.floor(pixels.length*.97))){const i=(p.y*W+p.x)*4; ed.data[i]=236;ed.data[i+1]=72;ed.data[i+2]=153;ed.data[i+3]=230;}
      for(const p of flame){const i=(p.y*W+p.x)*4; ed.data[i]=249;ed.data[i+1]=115;ed.data[i+2]=22;ed.data[i+3]=255;}
      for(const p of drop){const i=(p.y*W+p.x)*4; ed.data[i]=6;ed.data[i+1]=182;ed.data[i+2]=212;ed.data[i+3]=255;}
      ex.putImageData(ed,0,0);

      // Reproduce flameShapeGate's exact Lab threshold and morphology for the no-sample branch.
      let src=cv.imread(cvs), resized=new cv.Mat(); cv.resize(src,resized,new cv.Size(W,H)); src.delete();
      let bgr=new cv.Mat(),lab=new cv.Mat();cv.cvtColor(resized,bgr,cv.COLOR_RGBA2BGR);cv.cvtColor(bgr,lab,cv.COLOR_BGR2Lab);bgr.delete();resized.delete();
      const R=Math.round(g.r*.92),R2=R*R, vals=[], ld=lab.data;
      for(let y=Math.max(0,g.cy-R);y<Math.min(H,g.cy+R);y++)for(let x=Math.max(0,g.cx-R);x<Math.min(W,g.cx+R);x++){
        const dx=x-g.cx,dy=y-g.cy;if(dx*dx+dy*dy>R2)continue;const i=(y*W+x)*3,a=ld[i+1]-128,b=ld[i+2]-128;vals.push(Math.sqrt(a*a+b*b));
      }
      vals.sort((a,b)=>a-b);const threshold=Math.max(16,vals[Math.floor(vals.length*.88)]);
      let mask=cv.Mat.zeros(H,W,cv.CV_8UC1),md=mask.data;
      for(let y=Math.max(0,g.cy-R);y<Math.min(H,g.cy+R);y++)for(let x=Math.max(0,g.cx-R);x<Math.min(W,g.cx+R);x++){
        const dx=x-g.cx,dy=y-g.cy;if(dx*dx+dy*dy>R2)continue;const i=(y*W+x)*3,a=ld[i+1]-128,b=ld[i+2]-128,c=Math.sqrt(a*a+b*b);if(c>threshold&&b>4&&a>-25)md[y*W+x]=255;
      }
      lab.delete();let k3=cv.Mat.ones(3,3,cv.CV_8U),k7=cv.Mat.ones(7,7,cv.CV_8U);cv.morphologyEx(mask,mask,cv.MORPH_OPEN,k3);cv.morphologyEx(mask,mask,cv.MORPH_CLOSE,k7);k3.delete();k7.delete();
      const gate=canvas(),gd=opaqueBlack(gate.getContext('2d').createImageData(W,H));for(let i=0;i<md.length;i++)if(md[i]){gd.data[4*i]=255;gd.data[4*i+1]=255;gd.data[4*i+2]=255;} gate.getContext('2d').putImageData(gd,0,0);mask.delete();

      return {
        images:[original,encode(roi,.90),encode(chroma,.86),encode(gate,.90),encode(score,.86),encode(extracted,.90)],
        meta:{file:'005.png',state:result.state,h2:result.h2,rh:result.rh,roiSource:g.roiSource,cx:g.cx,cy:g.cy,r:g.r,innerR:g.innerR,nPx:g.nPx,nBg:g.nBg,nTop:g.nTop,nFlame:flame.length,nDrop:drop.length,chromaMin:cLo,chromaMax:cHi,shapeThreshold:6,gateThreshold:threshold,registration:reg}
      };
    });

    const [orig, roi, chroma, gate, score, extracted] = figureData.images;
    const m = figureData.meta;
    const fragment = `
<div id="actual-roi-pipeline">
  <style>
    #actual-roi-pipeline{color:var(--foreground);font-family:inherit;width:100%}
    #actual-roi-pipeline h2{margin:0 0 6px;font-weight:500}
    #actual-roi-pipeline .sub{margin:0 0 16px;color:var(--muted-foreground)}
    #actual-roi-pipeline .flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px 38px}
    #actual-roi-pipeline figure{margin:0;position:relative;min-width:0}
    #actual-roi-pipeline figure:not(:nth-child(3n))::after{content:'→';position:absolute;right:-29px;top:42%;color:var(--muted-foreground);font-size:26px}
    #actual-roi-pipeline img{width:100%;display:block;aspect-ratio:1;border:1px solid var(--border);object-fit:cover;background:var(--background)}
    #actual-roi-pipeline figcaption{padding-top:7px;font-weight:500}
    #actual-roi-pipeline .detail{display:block;color:var(--muted-foreground);font-weight:400;margin-top:2px}
    #actual-roi-pipeline .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:15px;color:var(--muted-foreground)}
    #actual-roi-pipeline .key{display:inline-flex;gap:6px;align-items:center}
    #actual-roi-pipeline .sw{width:11px;height:11px;display:inline-block}
    #actual-roi-pipeline .orange{background:var(--orange)}#actual-roi-pipeline .cyan{background:var(--blue)}#actual-roi-pipeline .pink{background:var(--purple)}#actual-roi-pipeline .gray{background:var(--muted-foreground)}
    @media(max-width:650px){#actual-roi-pipeline .flow{grid-template-columns:1fr;gap:22px}#actual-roi-pipeline figure::after{display:none}}
  </style>
  <h2>Actual ROI and feature extraction in the current app</h2>
  <p class="sub">Current app processing of 005.png · ${esc(m.state)} · ${esc(m.roiSource)} · ROI center (${m.cx}, ${m.cy}), r = ${m.r}px</p>
  <div class="flow">
    <figure><img alt="Original smartphone photograph" src="${orig}"><figcaption><b>a</b> Original image<span class="detail">Smartphone camera input</span></figcaption></figure>
    <figure><img alt="Detected outer and inner circular ROI with feature zones" src="${roi}"><figcaption><b>b</b> Circular ROI<span class="detail">red: r · yellow: 0.90r · dashed: feature zones</span></figcaption></figure>
    <figure><img alt="Corrected Lab chroma heat map" src="${chroma}"><figcaption><b>c</b> Corrected Lab chroma<span class="detail">C* after neutral-background correction</span></figcaption></figure>
    <figure><img alt="Binary flame silhouette used for sample presence gate" src="${gate}"><figcaption><b>d</b> Sample-presence mask<span class="detail">C* &gt; max(16, P88) + warm-color gate</span></figcaption></figure>
    <figure><img alt="Background-relative Lab distance heat map in flame and droplet zones" src="${score}"><figcaption><b>e</b> Shape score<span class="detail">ΔLab from bottom-50% chroma background</span></figcaption></figure>
    <figure><img alt="Extracted background top chroma flame and droplet pixels" src="${extracted}"><figcaption><b>f</b> Extracted pixels<span class="detail">top 35% per zone, ΔLab ≥ 6</span></figcaption></figure>
  </div>
  <div class="legend" aria-label="Extracted pixel colors"><span class="key"><i class="sw gray"></i>background 50%</span><span class="key"><i class="sw pink"></i>top chroma 3%</span><span class="key"><i class="sw orange"></i>flame (${m.nFlame.toLocaleString()} px)</span><span class="key"><i class="sw cyan"></i>registered droplet (${m.nDrop.toLocaleString()} px)</span></div>
</div>`;

    const fragmentPath = path.join(vizOut, 'actual-roi-extraction.html');
    fs.writeFileSync(fragmentPath, fragment, 'utf8');

    const standalone = `<!doctype html><meta charset="utf-8"><style>:root{--foreground:#17202a;--muted-foreground:#586574;--border:#cbd5e1;--background:#fff;--orange:#f97316;--blue:#06b6d4;--purple:#ec4899}body{margin:28px;background:white;font-family:Arial,sans-serif}#actual-roi-pipeline{max-width:1420px;margin:auto}</style>${fragment}`;
    const htmlPath = path.join(paperOut, 'actual-roi-extraction.html');
    fs.writeFileSync(htmlPath, standalone, 'utf8');
    fs.writeFileSync(path.join(paperOut, 'actual-roi-extraction-metadata.json'), JSON.stringify(m, null, 2), 'utf8');

    const outPage = await browser.newPage({ viewport: { width: 1500, height: 1100 }, deviceScaleFactor: 2 });
    await outPage.goto('file:///' + htmlPath.replaceAll('\\','/'));
    await outPage.locator('#actual-roi-pipeline').screenshot({ path: path.join(paperOut, 'actual-roi-extraction.png') });
    await outPage.pdf({ path: path.join(paperOut, 'actual-roi-extraction.pdf'), printBackground: true, width:'1500px', height:'1100px', margin:{top:'20px',right:'20px',bottom:'20px',left:'20px'} });
    await outPage.close();
    console.log(JSON.stringify({ fragmentPath, paperOut, metadata: m }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
