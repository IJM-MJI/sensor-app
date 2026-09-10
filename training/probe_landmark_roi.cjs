const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
const project='C:/Users/Administrator/Documents/ChatGPT/New project/sensor-app';
const source='D:/app_test/꼬리치레';
const opencv='C:/Users/Administrator/Documents/ChatGPT/New project/.tmp/deployed-sensor/opencv.js';
const files=['calibration.png','001.png','002.png','003.png','004.png','005.png','006.png','007.png','008.png','calibration_2.jpg','009.jpg','010.jpg','011.jpg','012.jpg','013jpg.jpg','014.jpg'];
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-background-networking','--host-resolver-rules=MAP * ~NOTFOUND']});
 try{
  const page=await browser.newPage({viewport:{width:480,height:900}});page.setDefaultTimeout(60000);
  await page.route('**/*',r=>{const u=new URL(r.request().url());if(u.href==='https://docs.opencv.org/4.9.0/opencv.js')return r.fulfill({path:opencv,contentType:'application/javascript'});const rel=u.pathname.slice(1)||'index.html';if(u.origin!=='https://local.sensor.test'||!/^(index\.html|sensor-[\w-]+\.js)$/.test(rel))return r.abort();return r.fulfill({path:path.join(project,rel),contentType:rel.endsWith('.html')?'text/html':'application/javascript'});});
  await page.goto('https://local.sensor.test/');await page.waitForFunction(()=>typeof cv!=='undefined'&&cv.Mat&&document.getElementById('loadingScreen').classList.contains('hidden'));
  const out=[];
  for(const file of files){
   await page.evaluate(()=>{calibrated=true;calMode=false;captured=false;});
   await page.locator('#testImageInput').setInputFiles(path.join(source,file));
   await page.waitForFunction(()=>captured&&document.getElementById('procOverlay').classList.contains('hidden'));
   const probe=await page.evaluate(()=>{
    let src=cv.imread(cvs),sc=Math.min(1,480/Math.max(src.cols,src.rows)),im=new cv.Mat();cv.resize(src,im,new cv.Size(Math.round(src.cols*sc),Math.round(src.rows*sc)));src.delete();
    let bgr=new cv.Mat(),lab=new cv.Mat();cv.cvtColor(im,bgr,cv.COLOR_RGBA2BGR);cv.cvtColor(bgr,lab,cv.COLOR_BGR2Lab);bgr.delete();im.delete();const w=lab.cols,h=lab.rows,d=lab.data;
    const vals=[];for(let y=Math.floor(h*.05);y<h*.68;y++)for(let x=Math.floor(w*.18);x<w*.82;x++){const i=(y*w+x)*3,a=d[i+1]-128,b=d[i+2]-128;if(b>2&&a>-35)vals.push(Math.sqrt(a*a+b*b));}vals.sort((a,b)=>a-b);
    const result={w,h,thresholds:{}};
    for(const [name,q,min] of [['p70',.70,9],['p80',.80,12],['p88',.88,16]]){
      const th=Math.max(min,vals[Math.floor(vals.length*q)]),mask=cv.Mat.zeros(h,w,cv.CV_8UC1),md=mask.data;
      for(let y=Math.floor(h*.05);y<h*.68;y++)for(let x=Math.floor(w*.18);x<w*.82;x++){const i=(y*w+x)*3,a=d[i+1]-128,b=d[i+2]-128,c=Math.sqrt(a*a+b*b);if(c>th&&b>2&&a>-35)md[y*w+x]=255;}
      const k3=cv.Mat.ones(3,3,cv.CV_8U),k5=cv.Mat.ones(5,5,cv.CV_8U);cv.morphologyEx(mask,mask,cv.MORPH_OPEN,k3);cv.morphologyEx(mask,mask,cv.MORPH_CLOSE,k5);k3.delete();k5.delete();
      const labels=new cv.Mat(),stats=new cv.Mat(),cents=new cv.Mat(),n=cv.connectedComponentsWithStats(mask,labels,stats,cents,8,cv.CV_32S),parts=[];
      for(let j=1;j<n;j++){const area=stats.intAt(j,4);if(area<25)continue;parts.push({area,x:stats.intAt(j,0),y:stats.intAt(j,1),w:stats.intAt(j,2),h:stats.intAt(j,3),cx:+cents.doubleAt(j,0).toFixed(1),cy:+cents.doubleAt(j,1).toFixed(1)});}
      parts.sort((a,b)=>b.area-a.area);result.thresholds[name]={th:+th.toFixed(1),parts:parts.slice(0,12)};labels.delete();stats.delete();cents.delete();mask.delete();
    }
    lab.delete();return result;
   });
   out.push({file,...probe});console.log(file,JSON.stringify(probe.thresholds.p80));
   await page.evaluate(()=>{retake();});
  }
  fs.writeFileSync(path.join(project,'training','landmark-probe.json'),JSON.stringify(out,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
