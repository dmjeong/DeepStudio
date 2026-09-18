import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
export type MaskHandle = { png: () => string };
export const MaskEditor = forwardRef<MaskHandle, {image: string; initialMask?: string; label?: string}>(function MaskEditor({image, initialMask, label = '불량 영역 편집'}, ref) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({width: 1, height: 1});
  const [ready, setReady] = useState(false), [error, setError] = useState(''), [tool, setTool] = useState('brush');
  const [brush, setBrush] = useState(12), [zoom, setZoom] = useState(1);
  const [width, setWidth] = useState(60), [height, setHeight] = useState(12), [angle, setAngle] = useState(0);
  const drawing = useRef(false), start = useRef<[number,number]>([0,0]), polygon = useRef<[number,number][]>([]), backup = useRef<ImageData | null>(null);
  useImperativeHandle(ref, () => ({png: () => {
    if (!ready || !canvas.current) throw new Error('이미지와 마스크를 불러오는 중입니다');
    const src = canvas.current, out = document.createElement('canvas');
    out.width = src.width; out.height = src.height;
    const ctx = out.getContext('2d')!;ctx.fillStyle = '#000';ctx.fillRect(0,0,out.width,out.height);ctx.drawImage(src,0,0);
    return out.toDataURL('image/png');
  }}), [ready]);
  useEffect(() => {
    let live = true;setReady(false);setError('');polygon.current = [];
    const img = new Image();
    img.onload = () => {
      if (!live || !canvas.current) return;
      const c = canvas.current;c.width = img.naturalWidth;c.height = img.naturalHeight;
      setSize({width:c.width,height:c.height});setZoom(Math.min(1,780/c.width));
      if (!initialMask) {setReady(true);return;}
      const mask = new Image();
      mask.onload = () => {
        if (!live) return;
        if(mask.naturalWidth!==c.width||mask.naturalHeight!==c.height){setError('마스크 크기가 원본과 다릅니다');return;}
        const ctx=c.getContext('2d')!;ctx.drawImage(mask,0,0);const data=ctx.getImageData(0,0,c.width,c.height);
        for(let n=0;n<data.data.length;n+=4){const alpha=data.data[n]>127?255:0;data.data[n]=data.data[n+1]=data.data[n+2]=255;data.data[n+3]=alpha;}
        ctx.putImageData(data,0,0);setReady(true);
      };
      mask.onerror=()=>{if(live)setError('마스크를 불러오지 못했습니다');};mask.src=initialMask;
    };
    img.onerror=()=>{if(live)setError('이미지를 불러오지 못했습니다');};img.src=image;
    return()=>{live=false;};
  },[image,initialMask]);
  const point=(e:React.PointerEvent<HTMLCanvasElement>):[number,number]=>{const r=e.currentTarget.getBoundingClientRect();return [Math.max(0,Math.min(size.width,(e.clientX-r.left)*size.width/r.width)),Math.max(0,Math.min(size.height,(e.clientY-r.top)*size.height/r.height))];};
  const context=()=>{const ctx=canvas.current!.getContext('2d')!;ctx.fillStyle=ctx.strokeStyle='#fff';ctx.lineWidth=brush;ctx.lineCap=ctx.lineJoin='round';return ctx;};
  const down=(e:React.PointerEvent<HTMLCanvasElement>)=>{
    if(!ready)return;const ctx=context(),p=point(e);backup.current=ctx.getImageData(0,0,size.width,size.height);
    if(tool==='polygon'){polygon.current.push(p);ctx.beginPath();ctx.arc(...p,Math.max(1,2/zoom),0,Math.PI*2);ctx.fill();return;}
    if(tool==='stamp'){ctx.save();ctx.translate(...p);ctx.rotate(angle*Math.PI/180);ctx.fillRect(-width/2,-height/2,width,height);ctx.restore();return;}
    drawing.current=true;start.current=p;e.currentTarget.setPointerCapture(e.pointerId);ctx.globalCompositeOperation=tool==='erase'?'destination-out':'source-over';ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(p[0]+.01,p[1]+.01);if(tool!=='rect')ctx.stroke();
  };
  const move=(e:React.PointerEvent<HTMLCanvasElement>)=>{if(!drawing.current)return;const ctx=context(),p=point(e);if(tool==='rect'){ctx.putImageData(backup.current!,0,0);ctx.fillRect(start.current[0],start.current[1],p[0]-start.current[0],p[1]-start.current[1]);}else{ctx.lineTo(...p);ctx.stroke();}};
  const up=()=>{drawing.current=false;if(canvas.current)canvas.current.getContext('2d')!.globalCompositeOperation='source-over';};
  return <div className="dg-editor"><div className="toolbar"><strong>{label}</strong><select aria-label="영역 도구" value={tool} onChange={e=>{setTool(e.target.value);polygon.current=[];}}><option value="brush">브러시</option><option value="erase">지우개</option><option value="rect">사각형</option><option value="polygon">다각형</option><option value="stamp">크기와 방향 지정</option></select><label>브러시 px <input type="number" min={1} max={500} value={brush} onChange={e=>setBrush(Math.max(1,+e.target.value))}/></label><button type="button" onClick={()=>{const c=canvas.current;if(c){c.getContext('2d')!.clearRect(0,0,c.width,c.height);polygon.current=[];}}}>영역 지우기</button><button type="button" onClick={()=>{if(backup.current)context().putImageData(backup.current,0,0);}}>실행 취소</button>{tool==='polygon'&&<button type="button" onClick={()=>{const points=polygon.current;if(points.length<3)return;const ctx=context();ctx.beginPath();ctx.moveTo(...points[0]);points.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();ctx.fill();polygon.current=[];}}>다각형 완료</button>}<label>확대 <input aria-label="확대" type="range" min={.1} max={4} step={.1} value={zoom} onChange={e=>setZoom(+e.target.value)}/></label></div>
    {tool==='stamp'&&<div className="toolbar"><label>길이 px <input type="number" min={1} value={width} onChange={e=>setWidth(Math.max(1,+e.target.value))}/></label><label>폭 px <input type="number" min={1} value={height} onChange={e=>setHeight(Math.max(1,+e.target.value))}/></label><label>회전 ° <input type="number" value={angle} onChange={e=>setAngle(+e.target.value)}/></label><span>이미지에서 중심을 클릭하세요</span></div>}
    {error&&<p role="alert">{error}</p>}<div className="dg-viewport"><div className="dg-canvas-wrap" style={{width:size.width*zoom,height:size.height*zoom}}><img alt={label} src={image} draggable={false}/><canvas ref={canvas} aria-label={label} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}/></div></div><small>{size.width} × {size.height} px | 스크롤로 이동, 확대 후에도 원본 좌표로 저장</small></div>;
});
export function Comparison({original,result}:{original:string;result:string}){
  const [split,setSplit]=useState(50),[zoom,setZoom]=useState(1);
  return <div><div className="toolbar"><label>원본 / 생성 비교 <input aria-label="원본 생성 비교" type="range" min={0} max={100} value={split} onChange={e=>setSplit(+e.target.value)}/></label><label>확대 <input type="range" min={1} max={4} step={.1} value={zoom} onChange={e=>setZoom(+e.target.value)}/></label></div><div className="dg-viewport"><div className="dg-compare" style={{width:`${zoom*100}%`}}><img alt="원본" src={original}/><img alt="생성 결과" src={result} style={{clipPath:`inset(0 0 0 ${split}%)`}}/><span style={{left:`${split}%`}}/></div></div></div>;
}
