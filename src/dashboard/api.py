"""
Dashboard web para RAPIRO Guardián.
Corre en la EC2 vía Docker. Lee eventos desde DynamoDB usando el
IAM instance profile (sin credenciales hardcodeadas).

Endpoints:
  GET /                        → dashboard HTML (tabs: Estadísticas + Documentos)
  GET /api/stats               → distribución de clases en JSON
  GET /api/sessions            → últimos 50 eventos en JSON
  GET /api/documents           → lista de documentos en S3 (incluye cuál es el activo)
  POST /upload                 → sube un PDF/txt a S3
  POST /api/documents/activate → marca un documento como activo (escribe active.txt)
  DELETE /api/documents        → elimina un documento de S3 (?name=filename.pdf)
  GET /health                  → liveness check
"""

import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RAPIRO Dashboard")

TABLE  = os.environ.get("DYNAMODB_TABLE", "rapiro_sessions_dev")
REGION = os.environ.get("AWS_DEFAULT_REGION", "sa-east-1")
BUCKET = os.environ.get("S3_BUCKET", "")

_dynamodb = boto3.resource("dynamodb", region_name=REGION)
_table    = _dynamodb.Table(TABLE)
_s3       = boto3.client("s3", region_name=REGION)

CLASS_LABELS   = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacío"}
_ALLOWED_TYPES = {"application/pdf", "text/plain"}
_MAX_BYTES     = 10 * 1024 * 1024  # 10 MB
_ACTIVE_KEY    = "documents/active.txt"


class ActivateRequest(BaseModel):
    name: str  # nombre del archivo sin prefijo "documents/"


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "table": TABLE, "region": REGION, "bucket": BUCKET}


@app.get("/api/sessions")
def get_sessions():
    resp  = _table.scan(Limit=50)
    items = sorted(
        resp.get("Items", []),
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )
    return {"sessions": items, "count": len(items)}


@app.get("/api/stats")
def get_stats():
    resp   = _table.scan()
    items  = resp.get("Items", [])
    counts = {0: 0, 1: 0, 2: 0}
    for item in items:
        c = item.get("clase_detectada")
        if c is not None:
            c = int(c)
            if c not in (0, 1, 2):
                c = 0  # mapear 3/4 → Estudiando
            counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    return {
        "total_events": total,
        "by_class": {CLASS_LABELS[k]: v for k, v in counts.items()},
        "percentages": {
            CLASS_LABELS[k]: round(v / total * 100, 1) if total else 0
            for k, v in counts.items()
        },
    }


@app.get("/api/documents")
def list_documents():
    if not BUCKET:
        return {"documents": [], "error": "S3_BUCKET no configurado"}
    try:
        active_name = ""
        try:
            body = _s3.get_object(Bucket=BUCKET, Key=_ACTIVE_KEY)["Body"].read()
            active_name = body.decode("utf-8", errors="replace").strip()
        except ClientError:
            pass

        resp = _s3.list_objects_v2(Bucket=BUCKET, Prefix="documents/")
        docs = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key in ("documents/", _ACTIVE_KEY):
                continue
            name = key.replace("documents/", "", 1)
            docs.append({
                "key": key,
                "name": name,
                "size_kb": round(obj["Size"] / 1024, 1),
                "last_modified": obj["LastModified"].isoformat(),
                "active": name == active_name,
            })
        return {"documents": docs, "count": len(docs), "active": active_name}
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/documents/activate")
def activate_document(req: ActivateRequest):
    if not BUCKET:
        raise HTTPException(status_code=503, detail="S3_BUCKET no configurado")
    key = f"documents/{req.name}"
    try:
        _s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError:
        raise HTTPException(status_code=404, detail=f"Archivo '{req.name}' no encontrado en S3")
    try:
        _s3.put_object(
            Bucket=BUCKET,
            Key=_ACTIVE_KEY,
            Body=req.name.encode("utf-8"),
            ContentType="text/plain",
        )
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ok": True, "active": req.name}


@app.delete("/api/documents")
def delete_document(name: str):
    if not BUCKET:
        raise HTTPException(status_code=503, detail="S3_BUCKET no configurado")
    key = f"documents/{name}"
    try:
        _s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError:
        raise HTTPException(status_code=404, detail=f"Archivo '{name}' no encontrado en S3")
    try:
        _s3.delete_object(Bucket=BUCKET, Key=key)
        # Si era el activo, borrar active.txt también
        try:
            body = _s3.get_object(Bucket=BUCKET, Key=_ACTIVE_KEY)["Body"].read()
            if body.decode("utf-8", errors="replace").strip() == name:
                _s3.delete_object(Bucket=BUCKET, Key=_ACTIVE_KEY)
        except ClientError:
            pass
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ok": True, "deleted": name}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not BUCKET:
        raise HTTPException(status_code=503, detail="S3_BUCKET no configurado")
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Solo se aceptan PDF o texto plano")
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 10 MB)")
    key = f"documents/{file.filename}"
    try:
        _s3.put_object(Bucket=BUCKET, Key=key, Body=content, ContentType=file.content_type)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ok": True, "key": key, "name": file.filename}


# ── HTML Dashboard ────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAPIRO Guardian</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
header{background:#1e293b;padding:1.5rem 2rem;border-bottom:1px solid #334155}
header h1{font-size:1.5rem;color:#38bdf8}
header p{color:#94a3b8;font-size:.875rem;margin-top:.25rem}
.tabs{display:flex;gap:0;background:#1e293b;border-bottom:1px solid #334155;padding:0 2rem}
.tab{padding:.75rem 1.5rem;cursor:pointer;color:#94a3b8;border-bottom:2px solid transparent;font-size:.875rem;transition:all .2s}
.tab.active{color:#38bdf8;border-bottom-color:#38bdf8}
.tab:hover:not(.active){color:#cbd5e1}
.panel{display:none}.panel.active{display:block}
main{max-width:1100px;margin:2rem auto;padding:0 1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1.5rem}
.card{background:#1e293b;border-radius:.75rem;padding:1.5rem;border:1px solid #334155}
.card .lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.card .val{font-size:2rem;font-weight:700;margin-top:.4rem}
.c0 .val{color:#22c55e}.c1 .val{color:#f59e0b}.c2 .val{color:#ef4444}.c3 .val{color:#38bdf8}.c4 .val{color:#a78bfa}.ct .val{color:#38bdf8}
.sec{background:#1e293b;border-radius:.75rem;border:1px solid #334155;margin-bottom:1.5rem}
.sec-h{padding:.75rem 1.5rem;border-bottom:1px solid #334155;font-weight:600;color:#cbd5e1;display:flex;align-items:center;justify-content:space-between}
table{width:100%;border-collapse:collapse}
th{padding:.6rem 1.5rem;text-align:left;font-size:.7rem;color:#64748b;text-transform:uppercase}
td{padding:.6rem 1.5rem;border-top:1px solid #1e293b;font-size:.85rem}
tr:hover td{background:#0f172a}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:9999px;font-size:.7rem;font-weight:600}
.b0{background:#14532d;color:#86efac}.b1{background:#78350f;color:#fcd34d}.b2{background:#7f1d1d;color:#fca5a5}.b3{background:#0c4a6e;color:#7dd3fc}.b4{background:#3b0764;color:#d8b4fe}
.bar-row{display:flex;align-items:center;gap:1rem;padding:.6rem 1.5rem}
.bar-lbl{width:140px;font-size:.85rem;color:#cbd5e1}
.bar-track{flex:1;background:#334155;border-radius:9999px;height:8px}
.bar-fill{height:8px;border-radius:9999px;transition:width .5s}
.bar-pct{width:45px;text-align:right;font-size:.85rem;color:#94a3b8}
.foot{text-align:center;color:#475569;font-size:.75rem;padding:1rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:.4rem;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* Upload */
.upload-zone{border:2px dashed #334155;border-radius:.75rem;padding:3rem 2rem;text-align:center;cursor:pointer;transition:border-color .2s;background:#0f172a;margin-bottom:1.5rem}
.upload-zone:hover,.upload-zone.drag{border-color:#38bdf8}
.upload-zone input{display:none}
.upload-zone .icon{font-size:2.5rem;margin-bottom:.75rem}
.upload-zone p{color:#94a3b8;font-size:.875rem}
.upload-zone .hint{color:#475569;font-size:.75rem;margin-top:.4rem}
.btn{display:inline-block;padding:.5rem 1.25rem;border-radius:.5rem;background:#38bdf8;color:#0f172a;font-weight:600;font-size:.875rem;cursor:pointer;border:none;margin-top:1rem}
.btn:hover{background:#7dd3fc}
.btn:disabled{background:#334155;color:#64748b;cursor:not-allowed}
.msg{padding:.6rem 1rem;border-radius:.5rem;font-size:.8rem;margin-bottom:.75rem}
.msg.ok{background:#14532d;color:#86efac}.msg.err{background:#7f1d1d;color:#fca5a5}
.doc-row td:first-child{font-family:monospace;font-size:.8rem}
.sz{color:#64748b;font-size:.75rem}
.act-badge{display:inline-block;padding:.15rem .5rem;border-radius:9999px;font-size:.65rem;font-weight:700;background:#134e4a;color:#5eead4;margin-left:.5rem;vertical-align:middle}
.btn-sm{padding:.25rem .65rem;border-radius:.375rem;font-size:.75rem;font-weight:600;cursor:pointer;border:none;margin-right:.3rem}
.btn-activate{background:#1e3a5f;color:#7dd3fc}.btn-activate:hover{background:#1e40af}
.btn-delete{background:#450a0a;color:#fca5a5}.btn-delete:hover{background:#7f1d1d}
.btn-activate.current{background:#134e4a;color:#5eead4;cursor:default}
</style>
</head>
<body>
<header>
  <h1>RAPIRO Guardian — Dashboard</h1>
  <p><span class="dot"></span>Asistente de estudio inteligente</p>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('stats')">Estadísticas</div>
  <div class="tab" onclick="switchTab('docs')">Documentos</div>
</div>

<!-- ── Tab Estadísticas ── -->
<div id="panel-stats" class="panel active">
<main>
  <div class="cards">
    <div class="card ct"><div class="lbl">Total eventos</div><div class="val" id="total">—</div></div>
    <div class="card c0"><div class="lbl">Estudiando</div><div class="val" id="p0">—</div></div>
    <div class="card c1"><div class="lbl">Usando celular</div><div class="val" id="p1">—</div></div>
    <div class="card c2"><div class="lbl">Puesto vacío</div><div class="val" id="p2">—</div></div>
  </div>
  <div class="sec">
    <div class="sec-h">Distribución por clase</div>
    <div id="bars"></div>
  </div>
  <div class="sec">
    <div class="sec-h">Últimos eventos</div>
    <table>
      <thead>
        <tr><th>Timestamp</th><th>Sesión</th><th>Estado</th><th>Confianza</th><th>Latencia</th></tr>
      </thead>
      <tbody id="tbody">
        <tr><td colspan="5" style="text-align:center;color:#475569;padding:2rem">Cargando...</td></tr>
      </tbody>
    </table>
  </div>
  <div class="foot" id="foot">Actualizando cada 10 segundos</div>
</main>
</div>

<!-- ── Tab Documentos ── -->
<div id="panel-docs" class="panel">
<main>
  <div id="upload-msg"></div>
  <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
       ondragover="event.preventDefault();this.classList.add('drag')"
       ondragleave="this.classList.remove('drag')"
       ondrop="handleDrop(event)">
    <input type="file" id="file-input" accept=".pdf,.txt" onchange="handleFile(this.files[0])">
    <div class="icon">📄</div>
    <p>Arrastra un PDF o haz clic para seleccionar</p>
    <p class="hint">PDF · TXT · máx 10 MB</p>
    <button class="btn" id="upload-btn" disabled onclick="event.stopPropagation();doUpload()">Subir documento</button>
  </div>
  <div class="sec">
    <div class="sec-h">
      <span>Documentos cargados</span>
      <span id="doc-count" style="color:#64748b;font-size:.75rem;font-weight:400"></span>
    </div>
    <table>
      <thead>
        <tr><th>Nombre</th><th>Tamaño</th><th>Fecha</th><th>Acciones</th></tr>
      </thead>
      <tbody id="doc-tbody">
        <tr><td colspan="4" style="text-align:center;color:#475569;padding:2rem">Cargando...</td></tr>
      </tbody>
    </table>
  </div>
</main>
</div>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name){
  document.querySelectorAll('.tab').forEach((t,i)=>{
    const n=i===0?'stats':'docs';
    t.classList.toggle('active',n===name);
  });
  document.querySelectorAll('.panel').forEach(p=>{
    p.classList.toggle('active',p.id==='panel-'+name);
  });
  if(name==='docs') loadDocs();
}

// ── Estadísticas ──────────────────────────────────────────────────────────────
const LABELS=["Estudiando","Usando celular","Puesto vacío"];
const COLORS=["#22c55e","#f59e0b","#ef4444"];
const BADGES=["b0","b1","b2"];

async function loadStats(){
  try{
    const r=await fetch("/api/stats");
    if(!r.ok){document.getElementById("total").textContent="HTTP "+r.status;return;}
    const d=await r.json();
    document.getElementById("total").textContent=d.total_events;
    const p=d.percentages||{};
    document.getElementById("p0").textContent=(p["Estudiando"]||0)+"%";
    document.getElementById("p1").textContent=(p["Usando celular"]||0)+"%";
    document.getElementById("p2").textContent=(p["Puesto vac\xEDo"]||0)+"%";
    document.getElementById("bars").innerHTML=LABELS.map((l,i)=>{
      const v=(p[l]||0);
      return '<div class="bar-row"><div class="bar-lbl">'+l+'</div>'+
             '<div class="bar-track"><div class="bar-fill" style="width:'+v+'%;background:'+COLORS[i]+'"></div></div>'+
             '<div class="bar-pct">'+v+'%</div></div>';
    }).join("");
  }catch(e){document.getElementById("total").textContent="ERR:"+e.message;}
}

async function loadSessions(){
  try{
    const r=await fetch("/api/sessions");
    if(!r.ok){
      document.getElementById("tbody").innerHTML=
        '<tr><td colspan="5" style="text-align:center;color:#ef4444;padding:2rem">HTTP '+r.status+'</td></tr>';
      return;
    }
    const d=await r.json();
    if(!d.sessions||!d.sessions.length){
      document.getElementById("tbody").innerHTML=
        '<tr><td colspan="5" style="text-align:center;color:#475569;padding:2rem">Sin eventos a\xFAn</td></tr>';
      return;
    }
    document.getElementById("tbody").innerHTML=d.sessions.slice(0,20).map(s=>{
      const c=Math.min(parseInt(s.clase_detectada)||0,2);
      const ts=(s.timestamp||"—").replace("T"," ").substring(0,19);
      const conf=s.confianza?(parseFloat(s.confianza)*100).toFixed(1)+"%":"—";
      const lat=s.latencia_ms?s.latencia_ms+" ms":"—";
      const sid=(s.session_id||"—").substring(0,8);
      return '<tr><td>'+ts+'</td>'+
             '<td style="font-family:monospace;font-size:.75rem">'+sid+'</td>'+
             '<td><span class="badge '+BADGES[c]+'">'+LABELS[c]+'</span></td>'+
             '<td>'+conf+'</td><td>'+lat+'</td></tr>';
    }).join("");
    document.getElementById("foot").textContent=
      "Actualizado: "+new Date().toLocaleTimeString("es-AR");
  }catch(e){
    document.getElementById("tbody").innerHTML=
      '<tr><td colspan="5" style="text-align:center;color:#ef4444;padding:2rem">ERR: '+e.message+'</td></tr>';
  }
}

function refreshStats(){loadStats();loadSessions();}
refreshStats();
setInterval(refreshStats,10000);

// ── Documentos ────────────────────────────────────────────────────────────────
let _selectedFile=null;

function handleFile(f){
  if(!f) return;
  _selectedFile=f;
  document.getElementById("upload-btn").disabled=false;
  document.getElementById("upload-btn").textContent="Subir: "+f.name;
}

function handleDrop(ev){
  ev.preventDefault();
  document.getElementById("drop-zone").classList.remove("drag");
  const f=ev.dataTransfer.files[0];
  if(f) handleFile(f);
}

async function doUpload(){
  if(!_selectedFile) return;
  const btn=document.getElementById("upload-btn");
  btn.disabled=true; btn.textContent="Subiendo...";
  const fd=new FormData();
  fd.append("file",_selectedFile);
  try{
    const r=await fetch("/upload",{method:"POST",body:fd});
    const j=await r.json();
    if(r.ok){
      showMsg("Documento '"+j.name+"' subido correctamente.","ok");
      _selectedFile=null;
      btn.textContent="Subir documento";
      loadDocs();
    } else {
      showMsg("Error: "+(j.detail||r.statusText),"err");
      btn.disabled=false; btn.textContent="Subir: "+_selectedFile.name;
    }
  }catch(e){
    showMsg("Error de red al subir el archivo.","err");
    btn.disabled=false; btn.textContent="Subir: "+_selectedFile.name;
  }
}

async function loadDocs(){
  try{
    const d=await(await fetch("/api/documents")).json();
    document.getElementById("doc-count").textContent=d.count+" archivo(s)"+(d.active?" · activo: "+d.active:"");
    if(!d.documents||!d.documents.length){
      document.getElementById("doc-tbody").innerHTML=
        '<tr><td colspan="4" style="text-align:center;color:#475569;padding:2rem">No hay documentos. Sube un PDF para que RAPIRO pueda responder sobre él.</td></tr>';
      return;
    }
    document.getElementById("doc-tbody").innerHTML=d.documents.map(doc=>{
      const dt=new Date(doc.last_modified).toLocaleString("es-AR");
      const activeBadge=doc.active?'<span class="act-badge">ACTIVO</span>':'';
      const activeBtnClass=doc.active?'btn-sm btn-activate current':'btn-sm btn-activate';
      const activeBtnLabel=doc.active?'✓ Activo':'Activar';
      const safeN=doc.name.replace(/&/g,'&amp;').replace(/"/g,'&quot;');
      return '<tr class="doc-row">'+
             '<td>'+doc.name+activeBadge+'</td>'+
             '<td class="sz">'+doc.size_kb+' KB</td>'+
             '<td class="sz">'+dt+'</td>'+
             '<td>'+
               '<button class="'+activeBtnClass+'" data-name="'+safeN+'" '+(doc.active?'disabled':'')+' onclick="activateDoc(this.dataset.name)">'+activeBtnLabel+'</button>'+
               '<button class="btn-sm btn-delete" data-name="'+safeN+'" onclick="deleteDoc(this.dataset.name)">Eliminar</button>'+
             '</td></tr>';
    }).join("");
  }catch(e){
    document.getElementById("doc-tbody").innerHTML=
      '<tr><td colspan="4" style="text-align:center;color:#475569;padding:2rem">Error al cargar documentos.</td></tr>';
  }
}

async function activateDoc(name){
  try{
    const r=await fetch("/api/documents/activate",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name})
    });
    const j=await r.json();
    if(r.ok){ showMsg("'"+name+"' ahora es el documento activo para RAPIRO.","ok"); loadDocs(); }
    else showMsg("Error: "+(j.detail||r.statusText),"err");
  }catch(e){ showMsg("Error de red.","err"); }
}

async function deleteDoc(name){
  if(!confirm("Eliminar '"+name+"' de S3?")) return;
  try{
    const r=await fetch("/api/documents?name="+encodeURIComponent(name),{method:"DELETE"});
    const j=await r.json();
    if(r.ok){ showMsg("'"+name+"' eliminado.","ok"); loadDocs(); }
    else showMsg("Error: "+(j.detail||r.statusText),"err");
  }catch(e){ showMsg("Error de red.","err"); }
}

function showMsg(text,type){
  const el=document.getElementById("upload-msg");
  el.innerHTML='<div class="msg '+type+'">'+text+'</div>';
  setTimeout(()=>{el.innerHTML="";},5000);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _HTML
