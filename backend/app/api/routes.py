import json, os, tempfile
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ..database.models import Project, Run, Document, MemoryItem
from ..memory.rag import ingest_document, retrieve
from ..schemas.agents import Influencer
from ..schemas.common import ProjectBrief
from ..services.reports import build_sections, stamp
from ..services.workflow import execute_run
from ..tools.influencer_search import export_csv
from ..tools.report_generator import build_pdf

router = APIRouter(prefix="/api")


def deps(request: Request):
    return request.app.state


def _project(db, pid):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    return p


# ---- projects -------------------------------------------------------------
@router.post("/projects")
def create_project(brief: ProjectBrief, st=Depends(deps)):
    with st.sessions() as db:
        p = Project(brief_json=brief.model_dump_json()); db.add(p); db.commit()
        return {"id": p.id}


@router.get("/projects")
def list_projects(st=Depends(deps)):
    with st.sessions() as db:
        out = []
        for p in db.query(Project).order_by(Project.id.desc()).all():
            last = p.runs[-1] if p.runs else None
            out.append({"id": p.id, **json.loads(p.brief_json), "created_at": p.created_at.isoformat(),
                        "last_run": {"id": last.id, "status": last.status} if last else None})
        return out


@router.get("/projects/{pid}")
def get_project(pid: int, st=Depends(deps)):
    with st.sessions() as db:
        p = _project(db, pid)
        return {"id": p.id, **json.loads(p.brief_json),
                "runs": [{"id": r.id, "status": r.status, "created_at": r.created_at.isoformat()} for r in p.runs]}


@router.post("/projects/{pid}/runs")
def start_run(pid: int, bg: BackgroundTasks, st=Depends(deps)):
    if st.llm is None or st.search is None:
        raise HTTPException(503, "LLM provider not configured (set LLM_API_KEY, or run Ollama locally)")
    with st.sessions() as db:
        _project(db, pid)
        r = Run(project_id=pid); db.add(r); db.commit(); rid = r.id
    bg.add_task(execute_run, st.sessions, rid, st.llm, st.search, 2, st.embedder)
    return {"run_id": rid}


# ---- runs -----------------------------------------------------------------
def _run(st, rid):
    with st.sessions() as db:
        r = db.get(Run, rid)
        if not r:
            raise HTTPException(404, "run not found")
        return r.status, r.result, json.loads(r.logs_json), json.loads(r.progress_json), r.project_id


@router.get("/runs")
def recent_runs(st=Depends(deps)):
    with st.sessions() as db:
        return [{"id": r.id, "project_id": r.project_id, "status": r.status, "created_at": r.created_at.isoformat(),
                 "company": json.loads(r.project.brief_json)["company_name"]}
                for r in db.query(Run).order_by(Run.id.desc()).limit(20).all()]


@router.get("/runs/{rid}")
def get_run(rid: int, st=Depends(deps)):
    status, result, logs, progress, pid = _run(st, rid)
    return {"status": status, "result": result, "logs": logs, "progress": progress, "project_id": pid}


class OutreachEdit(BaseModel):
    subject: str | None = None
    body: str | None = None
    approval_status: str | None = None   # draft | approved | rejected  (nothing is ever sent by the system)


@router.patch("/runs/{rid}/outreach/{idx}")
def edit_outreach(rid: int, idx: int, edit: OutreachEdit, st=Depends(deps)):
    if edit.approval_status not in (None, "draft", "approved", "rejected"):
        raise HTTPException(422, "approval_status must be draft, approved or rejected")
    with st.sessions() as db:
        r = db.get(Run, rid)
        if not r:
            raise HTTPException(404, "run not found")
        res = r.result
        drafts = res.get("influencer", {}).get("outreach", [])
        if not 0 <= idx < len(drafts):
            raise HTTPException(404, "draft not found")
        drafts[idx].update({k: v for k, v in edit.model_dump().items() if v is not None})
        r.result_json = json.dumps(res); db.commit()
        db.add(MemoryItem(project_id=r.project_id, kind="decision",
                          content=f"Outreach to {drafts[idx]['influencer_name']} ({drafts[idx]['channel']}): "
                                  f"{drafts[idx]['approval_status']}")); db.commit()
        return drafts[idx]


@router.get("/runs/{rid}/influencers.csv")
def influencers_csv(rid: int, st=Depends(deps)):
    result = _run(st, rid)[1]
    items = [Influencer(**i) for i in result.get("influencer", {}).get("influencers", [])]
    if not items:
        raise HTTPException(404, "no influencer results for this run")
    path = export_csv(items, os.path.join(tempfile.mkdtemp(), "influencers.csv"))
    return FileResponse(path, media_type="text/csv", filename="influencers.csv")


@router.get("/runs/{rid}/report.pdf")
def report_pdf(rid: int, section: str = "all", st=Depends(deps)):
    result = _run(st, rid)[1]
    try:
        title, sections = build_sections(result, section)
    except KeyError:
        raise HTTPException(404, "no completed section to report")
    path = build_pdf(os.path.join(tempfile.mkdtemp(), "report.pdf"), title, sections, stamp())
    return FileResponse(path, media_type="application/pdf", filename=f"marketmind-{section}.pdf")


# ---- knowledge / memory ---------------------------------------------------
@router.post("/projects/{pid}/documents")
async def upload_document(pid: int, file: UploadFile = File(...), st=Depends(deps)):
    data = await file.read()
    with st.sessions() as db:
        _project(db, pid)
        try:
            d = ingest_document(db, pid, file.filename or "upload", data, st.embedder)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"id": d.id, "filename": d.filename, "chunks": d.n_chunks}


@router.get("/projects/{pid}/documents")
def list_documents(pid: int, st=Depends(deps)):
    with st.sessions() as db:
        return [{"id": d.id, "filename": d.filename, "chunks": d.n_chunks}
                for d in db.query(Document).filter(Document.project_id == pid).all()]


class Query(BaseModel):
    query: str
    k: int = 5


@router.post("/projects/{pid}/knowledge/search")
def search_knowledge(pid: int, q: Query, st=Depends(deps)):
    with st.sessions() as db:
        return retrieve(db, pid, q.query, st.embedder, q.k)


class MemoryIn(BaseModel):
    kind: str = "note"
    content: str


@router.post("/projects/{pid}/memory")
def add_memory(pid: int, m: MemoryIn, st=Depends(deps)):
    with st.sessions() as db:
        _project(db, pid)
        item = MemoryItem(project_id=pid, kind=m.kind, content=m.content); db.add(item); db.commit()
        return {"id": item.id}


@router.get("/projects/{pid}/memory")
def list_memory(pid: int, st=Depends(deps)):
    with st.sessions() as db:
        return [{"id": m.id, "kind": m.kind, "content": m.content, "created_at": m.created_at.isoformat()}
                for m in db.query(MemoryItem).filter(MemoryItem.project_id == pid).order_by(MemoryItem.id.desc()).all()]
