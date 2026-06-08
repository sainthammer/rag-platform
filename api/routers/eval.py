"""POST /v1/eval/run — запустить RAGAS-оценку в фоне.
GET  /v1/eval/run/{job_id} — статус и результаты.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from api.jobs import get_job, new_job, update_job
from api.schemas import EvalJobResponse, EvalRunRequest, EvalStatusResponse

router = APIRouter(tags=["eval"])


async def _eval_task(job_id: str, req: EvalRunRequest) -> None:
    await update_job(job_id, "running")
    try:
        from evaluation.eval_runner import (
            _build_mock_pipeline,
            _build_ollama_pipeline,
            _build_ollama_ragas,
            generate_html_report,
            run_evaluation,
        )
        from evaluation.testcases_dataset import get_testcases

        if req.mode == "ollama":
            pipeline = _build_ollama_pipeline()
            ragas_llm, ragas_embeddings = _build_ollama_ragas()
        else:
            pipeline = _build_mock_pipeline()
            ragas_llm = ragas_embeddings = None

        testcases = get_testcases()
        result = await run_evaluation(
            pipeline=pipeline,
            testcases=testcases,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            max_ragas_cases=req.max_cases,
        )

        generate_html_report(
            result=result,
            mode=req.mode,
            output_path=req.output_path,
            max_ragas_cases=req.max_cases,
        )

        hall = result["hallucination_results"]
        await update_job(job_id, "done", {
            "report_path": str(Path(req.output_path).resolve()),
            "hallucination_pass": sum(1 for r in hall if r["passed"]),
            "hallucination_total": len(hall),
            "ragas_avg": result["ragas_avg"],
        })
    except Exception as exc:  # noqa: BLE001
        await update_job(job_id, "error", error=str(exc))


@router.post(
    "/eval/run",
    response_model=EvalJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить RAGAS-оценку пайплайна",
)
async def run_eval(req: EvalRunRequest) -> EvalJobResponse:
    """Запустить прогон тест-кейсов и RAGAS-оценку в фоновой задаче.

    Возвращает `job_id` для отслеживания через GET /v1/eval/run/{job_id}.
    """
    job = new_job()
    asyncio.create_task(_eval_task(job.job_id, req))
    return EvalJobResponse(job_id=job.job_id, status="pending")


@router.get(
    "/eval/run/{job_id}",
    response_model=EvalStatusResponse,
    summary="Статус и результаты оценки",
    responses={404: {"description": "Задача не найдена"}},
)
async def eval_status(job_id: str) -> EvalStatusResponse:
    """Вернуть статус задачи оценки. При status=done содержит путь к HTML-отчёту и средние баллы."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id!r} не найден")
    return EvalStatusResponse(
        job_id=job.job_id,
        status=job.status,
        report_path=job.result.get("report_path"),
        hallucination_pass=job.result.get("hallucination_pass"),
        hallucination_total=job.result.get("hallucination_total"),
        ragas_avg=job.result.get("ragas_avg"),
        error=job.error,
    )
