import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import {fileURLToPath} from "node:url";

const root=path.dirname(fileURLToPath(import.meta.url));
const meta=JSON.parse(fs.readFileSync(path.join(root,"trace_points.json"),"utf8"));
const cad={};
for(const kind of ["front","side","top"]){
 const svg=fs.readFileSync(path.join(root,meta.views[kind].svg),"utf8");
 const bytes=Buffer.from(svg,"utf8");
 const sha=crypto.createHash("sha1").update(Buffer.from("blob "+bytes.length+"\0","utf8")).update(bytes).digest("hex");
 if(sha!==meta.views[kind].blob)throw new Error("Source SVG changed: "+kind);
 const paths=[...svg.matchAll(/<path\s+d="([^"]+)"/g)].map(m=>[...m[1].matchAll(/[ML]\s*([-+\d.eE]+),([-+\d.eE]+)/g)].map(v=>[+v[1],+v[2]]));
 if(!paths.length)throw new Error("No CAD paths: "+kind);
 cad[kind]=paths.map(p=>p.map(([a,b])=>kind==="front"||kind==="side"?[b,a]:[a,-b]));
}
const norm={};
for(const kind of ["front","side","top"]){
 const t=meta[kind],[x0,y0,x1,y1]=t.bbox_px,[w,h]=t.target_dimensions_mm;
 norm[kind]=t.points_px.map(([x,y])=>[(x-x0)/(x1-x0)*w-(kind==="side"?0:w/2),kind==="top"?(y-y0)/(y1-y0)*h:h/2-(y-y0)/(y1-y0)*h]);
}

function plotEvidence(cad,norm) {
 const W=840,H=360, blue=[37,99,235],red=[220,38,38], gray=[232,236,242];
 const font={
 A:["01110","10001","10001","11111","10001","10001","10001"],B:["11110","10001","10001","11110","10001","10001","11110"],C:["01111","10000","10000","10000","10000","10000","01111"],D:["11110","10001","10001","10001","10001","10001","11110"],E:["11111","10000","10000","11110","10000","10000","11111"],F:["11111","10000","10000","11110","10000","10000","10000"],G:["01111","10000","10000","10111","10001","10001","01111"],H:["10001","10001","10001","11111","10001","10001","10001"],I:["11111","00100","00100","00100","00100","00100","11111"],J:["00111","00010","00010","00010","10010","10010","01100"],K:["10001","10010","10100","11000","10100","10010","10001"],L:["10000","10000","10000","10000","10000","10000","11111"],M:["10001","11011","10101","10101","10001","10001","10001"],N:["10001","11001","10101","10011","10001","10001","10001"],O:["01110","10001","10001","10001","10001","10001","01110"],P:["11110","10001","10001","11110","10000","10000","10000"],Q:["01110","10001","10001","10001","10101","10010","01101"],R:["11110","10001","10001","11110","10100","10010","10001"],S:["01111","10000","10000","01110","00001","00001","11110"],T:["11111","00100","00100","00100","00100","00100","00100"],U:["10001","10001","10001","10001","10001","10001","01110"],V:["10001","10001","10001","10001","10001","01010","00100"],W:["10001","10001","10001","10101","10101","10101","01010"],X:["10001","10001","01010","00100","01010","10001","10001"],Y:["10001","10001","01010","00100","00100","00100","00100"],Z:["11111","00001","00010","00100","01000","10000","11111"],
 "0":["01110","10001","10011","10101","11001","10001","01110"],"1":["00100","01100","00100","00100","00100","00100","01110"],"2":["01110","10001","00001","00010","00100","01000","11111"],"3":["11110","00001","00001","01110","00001","00001","11110"],"4":["00010","00110","01010","10010","11111","00010","00010"],"5":["11111","10000","10000","11110","00001","00001","11110"],"6":["01110","10000","10000","11110","10001","10001","01110"],"7":["11111","00001","00010","00100","01000","01000","01000"],"8":["01110","10001","10001","01110","10001","10001","01110"],"9":["01110","10001","10001","01111","00001","00001","01110"],"-":["00000","00000","00000","11111","00000","00000","00000"],".":["00000","00000","00000","00000","00000","00100","00100"],":":["00000","00100","00100","00000","00100","00100","00000"],"/":["00001","00001","00010","00100","01000","10000","10000"]
 };
 function b64(bytes){const c="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";let o=[];for(let i=0;i<bytes.length;i+=3){let v=(bytes[i]<<16)|((bytes[i+1]||0)<<8)|(bytes[i+2]||0);o.push(c[(v>>>18)&63]+c[(v>>>12)&63]+(i+1<bytes.length?c[(v>>>6)&63]:"=")+(i+2<bytes.length?c[v&63]:"="));}return o.join("");}
 function png(rgb,w,h){
   let raw=[];for(let y=0;y<h;y++){raw.push(0);for(let x=0;x<w*3;x++)raw.push(rgb[y*w*3+x]);}
   const out=[120,1];let buf=0,nbit=0;
   function bits(v,n){buf|=v<<nbit;nbit+=n;while(nbit>=8){out.push(buf&255);buf>>>=8;nbit-=8;}}
   function rev(v,n){let r=0;for(let i=0;i<n;i++){r=(r<<1)|(v&1);v>>>=1;}return r;}
   function sym(s){let v,n;if(s<=143){v=48+s;n=8;}else if(s<=255){v=400+s-144;n=9;}else if(s<=279){v=s-256;n=7;}else{v=192+s-280;n=8;}bits(rev(v,n),n);}
   const lb=[3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258],le=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
   bits(3,3);
   for(let i=0;i<raw.length;){
     let best=0,dist=0;
     for(let d of [1,3])if(i>=d){let n=0;while(n<258&&i+n<raw.length&&raw[i+n]===raw[i+n-d])n++;if(n>best){best=n;dist=d;}}
     if(best>=3){let k=0;while(k<28&&best>=lb[k+1])k++;sym(257+k);if(le[k])bits(best-lb[k],le[k]);bits(rev(dist===1?0:2,5),5);i+=best;}else{sym(raw[i++]);}
   }
   sym(256);if(nbit)out.push(buf&255);
   let a=1,b=0;for(let v of raw){a=(a+v)%65521;b=(b+a)%65521;}out.push((b>>>8)&255,b&255,(a>>>8)&255,a&255);
   const crcTable=[];for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?(0xedb88320^(c>>>1)):(c>>>1);crcTable[n]=c>>>0;}
   function u32(v){return [(v>>>24)&255,(v>>>16)&255,(v>>>8)&255,v&255];}
   function chunk(type,data){let t=[...type].map(c=>c.charCodeAt(0)),c=0xffffffff;for(let v of [...t,...data])c=crcTable[(c^v)&255]^(c>>>8);return [...u32(data.length),...t,...data,...u32((c^0xffffffff)>>>0)];}
   return b64([137,80,78,71,13,10,26,10,...chunk("IHDR",[...u32(w),...u32(h),8,2,0,0,0]),...chunk("IDAT",out),...chunk("IEND",[])]);
 }
 function make(kind,overlay){
   const rgb=new Uint8Array(W*H*3);rgb.fill(255);
   function pix(x,y,c,alpha=1){if(x<0||x>=W||y<0||y>=H)return;let i=(y*W+x)*3;for(let k=0;k<3;k++)rgb[i+k]=Math.round(rgb[i+k]*(1-alpha)+c[k]*alpha);}
   function line(a,b,c,width=2){let dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy,r=width/2;
     for(let y=Math.max(0,Math.floor(Math.min(a[1],b[1])-r-1));y<=Math.min(H-1,Math.ceil(Math.max(a[1],b[1])+r+1));y++)
       for(let x=Math.max(0,Math.floor(Math.min(a[0],b[0])-r-1));x<=Math.min(W-1,Math.ceil(Math.max(a[0],b[0])+r+1));x++){
         let t=den?Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/den)):0;
         let dist=Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy),aa=Math.max(0,Math.min(1,r+.6-dist));if(aa)pix(x,y,c,aa);
       }
   }
   function txt(s,x,y,color=[41,52,68],scale=2){for(let ch of s.toUpperCase()){let rows=font[ch];if(rows)for(let yy=0;yy<7;yy++)for(let xx=0;xx<5;xx++)if(rows[yy][xx]==="1")for(let sy=0;sy<scale;sy++)for(let sx=0;sx<scale;sx++)pix(x+xx*scale+sx,y+yy*scale+sy,color);x+=6*scale;}}
   const width=kind==="side"?(overlay?30:40):100,height=kind==="top"?(overlay?30:40):25,scale=kind==="top"&&!overlay?4.5:6,originX=kind==="side"?420-width*scale/2:420,originY=kind==="top"?123:220;
   const map=([x,y])=>[originX+x*scale,originY+(kind==="top"?y:-y)*scale];
   let svg=[];
   const ru={front:"Спереди · X–Z",side:"Сбоку · глубина–Z",top:"Сверху · X–глубина"}[kind];
   svg.push('<svg xmlns="http://www.w3.org/2000/svg" width="840" height="360" viewBox="0 0 840 360"><rect width="840" height="360" fill="white"/>');
   svg.push('<style>text{font-family:Arial,sans-serif;fill:#293444}</style>');
   svg.push('<text x="26" y="35" font-size="22" font-weight="bold">'+ru+' · '+width+' × '+height+' мм</text>');
   svg.push('<text x="26" y="60" font-size="14">'+(overlay?'Синий: CAD v3. Красный: ручная трассировка нового вложения №4.':'Ортогональные рёбра из SVG-экспорта handle.step; без перспективы.')+'</text>');
   svg.push('<text x="26" y="84" font-size="13">'+(overlay?'Глубина CAD 40→30 мм только на изображении; геометрия не изменена.':'Монтажный участок сборочной ручки в этом базовом STEP отсутствует.')+'</text>');
   txt((overlay?"OVERLAY ":"CAD ")+kind+"  "+width+" X "+height+" MM",26,23);
   txt(overlay?"BLUE: CAD V3   RED: NEW SKETCH - MANUAL TRACE":"ORTHOGRAPHIC STEP EDGES - NO PERSPECTIVE",26,49,[67,77,93],1);
   txt(overlay?"CAD DEPTH 40 TO 30 FOR SHAPE COMPARISON ONLY":"BASE HANDLE ONLY - MOUNTING LAND NOT INCLUDED",26,72,[67,77,93],1);
   for(let x=kind==="side"?0:-50;x<= (kind==="side"?width:50);x+=10){
     const a=map([x,kind==="top"?0:-height/2]),b=map([x,kind==="top"?height:height/2]);
     line(a,b,gray,1);svg.push('<path d="M'+a.join(",")+' L'+b.join(",")+'" stroke="#e8ecf2" fill="none"/>');
   }
   for(let y=kind==="top"?0:-10;y<= (kind==="top"?height:10);y+=5){
     const a=map([kind==="side"?0:-50,y]),b=map([kind==="side"?width:50,y]);
     line(a,b,gray,1);svg.push('<path d="M'+a.join(",")+' L'+b.join(",")+'" stroke="#e8ecf2" fill="none"/>');
   }
   for(let p of cad[kind]){
     let q=p.map(v=>[(kind==="side"&&overlay)?v[0]*.75:v[0],(kind==="top"&&overlay)?v[1]*.75:v[1]]).map(map);
     for(let j=1;j<q.length;j++)line(q[j-1],q[j],blue,1.65);
     svg.push('<path d="'+q.map((v,i)=>(i?"L":"M")+v.map(a=>a.toFixed(3)).join(",")).join(" ")+'" stroke="#2563eb" stroke-width="1.65" fill="none" stroke-linejoin="round"/>');
   }
   if(overlay){
     let q=norm[kind].map(map);q.push(q[0]);
     for(let j=1;j<q.length;j++)line(q[j-1],q[j],red,2.5);
     svg.push('<path d="'+q.map((v,i)=>(i?"L":"M")+v.map(a=>a.toFixed(3)).join(",")).join(" ")+' Z" stroke="#dc2626" stroke-width="2.5" fill="none"/>');
     for(let p of q.slice(0,-1)){line(p,p,red,5);svg.push('<circle cx="'+p[0].toFixed(3)+'" cy="'+p[1].toFixed(3)+'" r="2.5" fill="#dc2626"/>');}
   }
   txt(overlay?"TRACE UNCERTAINTY: 5-10 SOURCE PIXELS. CHECK ORIGINAL.":"SOURCE: 4D35629 - CORRECTED-WINDOW-V3/CAD/HANDLE.STEP",26,336,[95,105,120],1);
   svg.push('<text x="26" y="347" font-size="12">'+(overlay?'Точки соединены отрезками; погрешность чтения 5–10 пикселей. Сверить с оригиналом.':'Источник: 4d35629 · corrected-window-v3/cad/handle.step')+'</text></svg>');
   return {png:png(rgb,W,H),svg:svg.join("\n")};
 }
 let files={};for(let k of ["front","side","top"]){files["cad_"+k]=make(k,false);files["overlay_"+k]=make(k,true);}return files;
}

const result=plotEvidence(cad,norm);
for(const [name,data] of Object.entries(result)){
 fs.writeFileSync(path.join(root,name+".svg"),data.svg,"utf8");
 fs.writeFileSync(path.join(root,name+".png"),Buffer.from(data.png,"base64"));
}
console.log("Wrote three CAD projections and three manual-trace overlays, SVG and PNG.");
