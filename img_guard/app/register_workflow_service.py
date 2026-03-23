from __future__ import annotations

from typing import Any

from app.contracts_v1 import (
    ArchiveImageRequestV1,
    InputItemV1,
    RegisterWorkflowAssetsV1,
    RegisterWorkflowRequestV1,
    RegisterWorkflowResponseV1,
    VectorUpsertRequestV1,
    WatermarkOptionsV1,
)
from app.guard_service import run_guard_v1
from app.persist_service import archive_image_v1, upsert_vector_embedding_v1
from app.watermark.models import MediaInput, WatermarkEmbedOptions, WatermarkEmbedRequest
from app.watermark.service import WatermarkService


def _normalize_input(item: InputItemV1) -> dict[str, Any]:
    return item.model_dump(exclude_none=True)


def _build_guard_request(req: RegisterWorkflowRequestV1, input_item: InputItemV1) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": req.job_id,
        "mode": "register",
        "content_type": "image",
        "input": [_normalize_input(input_item)],
        "meta": req.meta,
    }
    if req.guard_options is not None:
        payload["options"] = req.guard_options.model_dump(exclude_none=True)
    return payload


def _to_media_input(item: InputItemV1) -> MediaInput:
    return MediaInput(
        url=item.url,
        s3_uri=item.s3_uri,
        s3_key=item.s3_key,
        local_path=item.local_path,
        filename=item.filename,
        mime_type=item.mime_type,
    )


def _to_wm_options(opt: WatermarkOptionsV1 | None) -> WatermarkEmbedOptions:
    if opt is None:
        return WatermarkEmbedOptions()
    return WatermarkEmbedOptions(
        model=opt.model or "wam",
        nbits=opt.nbits if opt.nbits is not None else 32,
        scaling_w=opt.scaling_w if opt.scaling_w is not None else 2.0,
        proportion_masked=opt.proportion_masked if opt.proportion_masked is not None else 0.65,
    )


def _to_s3_input_from_archive(
    archived_key: str | None,
    fallback_input: InputItemV1,
) -> InputItemV1:
    if archived_key:
        return InputItemV1(
            s3_key=archived_key,
            filename=fallback_input.filename,
            mime_type=fallback_input.mime_type,
        )
    return fallback_input


def run_register_workflow_v1(req: RegisterWorkflowRequestV1 | dict[str, Any]) -> RegisterWorkflowResponseV1:
    parsed = req if isinstance(req, RegisterWorkflowRequestV1) else RegisterWorkflowRequestV1.model_validate(req)

    assets = RegisterWorkflowAssetsV1()
    warnings: list[str] = []
    pending_actions: list[str] = []
    vector_upsert_resp = None
    watermark_embed_success: bool | None = None
    watermark_output_key: str | None = None

    source_input = parsed.input

    if parsed.options.archive_register_request:
        arch = archive_image_v1(
            ArchiveImageRequestV1(
                job_id=parsed.job_id,
                kind="register_request",
                input=source_input,
                meta=parsed.meta,
                bucket=parsed.bucket,
                output_filename=source_input.filename,
            )
        )
        if arch.success:
            assets.register_request_s3_key = arch.s3_key
            source_input = _to_s3_input_from_archive(arch.s3_key, source_input)
        else:
            warnings.append(f"archive(register_request) failed: {arch.reason}")

    guard_resp = run_guard_v1(_build_guard_request(parsed, source_input))
    decision = guard_resp.decision

    if decision == "block":
        if parsed.options.archive_rejected_request:
            rej = archive_image_v1(
                ArchiveImageRequestV1(
                    job_id=parsed.job_id,
                    kind="rejected_request",
                    input=source_input,
                    meta=parsed.meta,
                    bucket=parsed.bucket,
                    output_filename=source_input.filename,
                )
            )
            if rej.success:
                assets.rejected_request_s3_key = rej.s3_key
            else:
                warnings.append(f"archive(rejected_request) failed: {rej.reason}")

        return RegisterWorkflowResponseV1(
            job_id=parsed.job_id,
            success=True,
            decision=decision,
            next_action=guard_resp.next_action,
            reason=guard_resp.reason,
            guard=guard_resp,
            assets=assets,
            watermark_embed_success=watermark_embed_success,
            watermark_output_key=watermark_output_key,
            vector_upsert=vector_upsert_resp,
            pending_actions=pending_actions,
            warnings=warnings,
        )

    if decision == "review":
        pending_actions.append("start_vote")
        return RegisterWorkflowResponseV1(
            job_id=parsed.job_id,
            success=True,
            decision=decision,
            next_action=guard_resp.next_action,
            reason=guard_resp.reason,
            guard=guard_resp,
            assets=assets,
            watermark_embed_success=watermark_embed_success,
            watermark_output_key=watermark_output_key,
            vector_upsert=vector_upsert_resp,
            pending_actions=pending_actions,
            warnings=warnings,
        )

    wm_source_input = source_input
    if parsed.options.archive_wm_request_original:
        wm_req_arch = archive_image_v1(
            ArchiveImageRequestV1(
                job_id=parsed.job_id,
                kind="watermark_request_original",
                input=source_input,
                meta=parsed.meta,
                bucket=parsed.bucket,
                output_filename=source_input.filename,
            )
        )
        if wm_req_arch.success:
            assets.wm_request_original_s3_key = wm_req_arch.s3_key
            wm_source_input = _to_s3_input_from_archive(wm_req_arch.s3_key, source_input)
        else:
            warnings.append(f"archive(watermark_request_original) failed: {wm_req_arch.reason}")

    wm_service = WatermarkService.create()
    wm_resp = wm_service.embed(
        WatermarkEmbedRequest(
            job_id=parsed.job_id,
            input=_to_media_input(wm_source_input),
            meta=parsed.meta,
            options=_to_wm_options(parsed.watermark_options or (parsed.guard_options.watermark if parsed.guard_options else None)),
        )
    )
    watermark_embed_success = bool(wm_resp.success and wm_resp.result.applied)

    if not wm_resp.success:
        warnings.append(f"watermark embed failed: {wm_resp.reason}")

    if parsed.options.archive_wm_result and wm_resp.result.output_path:
        wm_res_arch = archive_image_v1(
            ArchiveImageRequestV1(
                job_id=parsed.job_id,
                kind="watermark_result",
                input=InputItemV1(local_path=wm_resp.result.output_path, filename=source_input.filename),
                meta=parsed.meta,
                bucket=parsed.bucket,
            )
        )
        if wm_res_arch.success:
            assets.wm_result_s3_key = wm_res_arch.s3_key
            watermark_output_key = wm_res_arch.s3_key
        else:
            warnings.append(f"archive(watermark_result) failed: {wm_res_arch.reason}")
    else:
        watermark_output_key = wm_resp.result.output_key

    if parsed.options.upsert_vector_on_allow:
        token_meta_key = parsed.options.token_issued_meta_key
        token_issued = bool(parsed.meta.get(token_meta_key, False))
        if parsed.options.require_token_issued_for_upsert and not token_issued:
            pending_actions.append("upsert_vector_after_token")
        else:
            upsert_input = _to_s3_input_from_archive(assets.wm_request_original_s3_key or assets.register_request_s3_key, source_input)
            vector_upsert_resp = upsert_vector_embedding_v1(
                VectorUpsertRequestV1(
                    job_id=parsed.job_id,
                    input=upsert_input,
                    s3_key=upsert_input.s3_key,
                    file_name=source_input.filename,
                )
            )
            if not vector_upsert_resp.success:
                warnings.append(f"vector upsert failed: {vector_upsert_resp.reason}")

    return RegisterWorkflowResponseV1(
        job_id=parsed.job_id,
        success=True,
        decision=decision,
        next_action=guard_resp.next_action,
        reason=guard_resp.reason,
        guard=guard_resp,
        assets=assets,
        watermark_embed_success=watermark_embed_success,
        watermark_output_key=watermark_output_key,
        vector_upsert=vector_upsert_resp,
        pending_actions=pending_actions,
        warnings=warnings,
    )

