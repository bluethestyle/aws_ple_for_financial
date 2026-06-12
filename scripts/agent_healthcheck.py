"""운영·감사 에이전트 헬스체크 (AWS) — 전체 스택이 정상 동작하는지 한 번에 점검.

온프렘 scripts/agent_healthcheck.py (d74f012c) 의 AWS 등가물.
Ollama / GPU 가드 항목은 온프렘 환경 특이라 제외하고 Bedrock 스택으로 치환했다.

사용:
  python scripts/agent_healthcheck.py          # 빠른 점검 (Bedrock 실호출 없음)
  python scripts/agent_healthcheck.py --full   # Bedrock converse 실호출 1회 포함

점검: AWS 자격증명 / Bedrock model id 유효성 / agent_tools.yaml↔tool_wrappers
      구현 정합 / agent_tool_routing.yaml 도구명 실존성 + focus×agents 교집합
      (PORT-02·silent no-op 회귀 방지) /
      case_store 백엔드 / pipeline_reports 임포트 / (--full) Bedrock 실호출
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

TOOLS_YAML = REPO_ROOT / "configs" / "financial" / "agent_tools.yaml"
ROUTING_YAML = REPO_ROOT / "configs" / "financial" / "agent_tool_routing.yaml"
PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"

results = []


def check(name, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"예외: {e}"
    dt = time.time() - t0
    mark = "✅" if ok else "❌"
    results.append(ok)
    print(f"{mark} {name:32} {detail}  ({dt:.1f}s)")


def _pipeline_cfg() -> dict:
    import yaml
    if PIPELINE_YAML.exists():
        return yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8")) or {}
    return {}


def _region():
    """pipeline.yaml → AWS_REGION env 순으로 해석. 모두 없으면 None 을 반환해
    boto3 기본 해석(프로파일/AWS_DEFAULT_REGION)에 위임한다 — 거기서도 못 찾으면
    NoRegionError 로 명시 실패 (region 하드코딩 폴백 금지)."""
    cfg = _pipeline_cfg()
    return (
        (cfg.get("aws") or {}).get("region")
        or ((cfg.get("llm_provider") or {}).get("bedrock") or {}).get("region")
        or os.getenv("AWS_REGION")
    )


def _configured_model_ids() -> dict:
    """pipeline.yaml llm_provider.bedrock 의 모든 model_id (config-driven)."""
    bedrock_cfg = ((_pipeline_cfg().get("llm_provider") or {}).get("bedrock") or {})
    ids = {}
    for agent, spec in (bedrock_cfg.get("models") or {}).items():
        if isinstance(spec, dict) and spec.get("model_id"):
            ids[agent] = spec["model_id"]
    default = (bedrock_cfg.get("default") or {}).get("model_id")
    if default:
        ids["default"] = default
    return ids


# ---------------------------------------------------------------------------
# 기본 점검 (AWS 호출은 control-plane 만, LLM 토큰 소비 없음)
# ---------------------------------------------------------------------------

def c_credentials():
    import boto3
    ident = boto3.client("sts", region_name=_region()).get_caller_identity()
    account = ident.get("Account", "?")
    return True, f"계정 {account[:4]}**** / region={_region() or '(boto3 기본 해석)'}"


def c_bedrock_models():
    """pipeline.yaml 의 model_id 가 foundation model 또는 inference profile 로 실존하는지."""
    import boto3
    ids = _configured_model_ids()
    if not ids:
        return False, "pipeline.yaml llm_provider.bedrock 에 model_id 없음"
    client = boto3.client("bedrock", region_name=_region())
    known = set()
    for fm in client.list_foundation_models().get("modelSummaries", []):
        known.add(fm.get("modelId", ""))
    try:
        paginator = client.get_paginator("list_inference_profiles")
        for page in paginator.paginate():
            for p in page.get("inferenceProfileSummaries", []):
                known.add(p.get("inferenceProfileId", ""))
    except Exception:
        pass  # 일부 region 은 inference profile API 미지원 — foundation 만으로 판정
    missing = {a: m for a, m in ids.items() if m not in known}
    if missing:
        return False, f"미존재 model_id: {missing}"
    return True, f"{len(ids)}개 model_id 확인 ({', '.join(sorted(set(ids.values())))})"


def c_tool_contract():
    """agent_tools.yaml 스키마 ↔ tool_wrappers 구현의 양방향 정합."""
    from core.agent.tool_registry import ToolRegistry
    from core.agent.tool_wrappers import register_all_tools

    reg = ToolRegistry(config_path=str(TOOLS_YAML) if TOOLS_YAML.exists() else None)
    schema_names = set(reg._tools)
    n = register_all_tools(reg)
    schema_only = sorted(s for s in schema_names if reg._tools[s].func is None)
    impl_only = sorted(set(reg._tools) - schema_names)
    ok = not schema_only and not impl_only
    detail = f"스키마 {len(schema_names)}개, 구현 {n}개"
    if schema_only:
        detail += f" | 스키마만 있고 미구현: {schema_only}"
    if impl_only:
        detail += f" | 구현만 있고 스키마 없음: {impl_only}"
    return ok, detail


def c_tool_routing():
    """agent_tool_routing.yaml + 내장 폴백의 도구명이 schema 에 실존하는지 (PORT-02)
    + agents 축: focus 를 실제 호출하는 agent 에게 tool_map 도구가 1개 이상
    노출되는지 (_build_tool_config 의 get_bedrock_tools 교집합에서 전부 탈락하면
    silent no-op — audit+herding 회귀 가드, 2026-06-12)."""
    import yaml
    from core.agent.bedrock_dialog import _ROUTING_DEFAULT
    from core.agent.pipeline_reports import _AUDIT_AREA_FOCUS_MAP

    schema = yaml.safe_load(TOOLS_YAML.read_text(encoding="utf-8"))
    schema_names = {t["name"] for t in schema.get("tools", [])}
    agents_of = {
        t["name"]: t.get("agents", ["ops", "audit"]) for t in schema.get("tools", [])
    }
    # 프로덕션 호출자: CP* 는 ops followup, _AUDIT_AREA_FOCUS_MAP 값들은 audit.
    audit_focuses = set(_AUDIT_AREA_FOCUS_MAP.values())

    def names_of(routing):
        out = set()
        for sec in ("curated", "tool_map", "common"):
            for tools in (routing.get(sec) or {}).values():
                out.update(tools)
        return out

    routings = [_ROUTING_DEFAULT]
    unknown = names_of(_ROUTING_DEFAULT) - schema_names
    n_yaml = 0
    if ROUTING_YAML.exists():
        routing = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8")) or {}
        yaml_names = names_of(routing)
        n_yaml = len(yaml_names)
        unknown |= yaml_names - schema_names
        routings.append(routing)
    if unknown:
        return False, f"schema 미존재 도구명: {sorted(unknown)}"
    noop = set()
    for r in routings:
        for focus, tools in (r.get("tool_map") or {}).items():
            caller = "audit" if focus in audit_focuses else "ops"
            if not any(caller in agents_of.get(t, []) for t in tools):
                noop.add(f"{focus}(agent={caller})")
    if noop:
        return False, f"agents 축 silent no-op focus: {sorted(noop)}"
    return True, f"routing 도구명 {n_yaml}개 실존 + focus×agents 교집합 OK"


def c_case_store():
    import shutil
    import tempfile
    from core.agent.case_store import DiagnosticCaseStore

    path = tempfile.mkdtemp(prefix="_hc_case_store_")
    try:
        cs = DiagnosticCaseStore(store_path=path)
        cs.save_case({
            "agent": "HC", "pipeline_part": "CP5", "verdict": "PASS",
            "finding": "healthcheck", "check_item": "healthcheck",
        })
        return True, f"backend={cs._backend}, save_case OK"
    finally:
        shutil.rmtree(path, ignore_errors=True)


def c_imports():
    import importlib
    mods = [
        "core.agent.pipeline_reports",
        "core.agent.bedrock_dialog",
        "core.agent.ops.collector",
        "core.agent.ops.diagnoser",
        "core.agent.ops.reporter",
        "core.agent.audit.diagnoser",
        "core.agent.audit.reporter",
        "core.agent.case_store",
    ]
    for m in mods:
        importlib.import_module(m)
    return True, f"{len(mods)}개 모듈 임포트 OK"


# ---------------------------------------------------------------------------
# 심화 점검 (--full): Bedrock converse 실호출 1회
# ---------------------------------------------------------------------------

def c_bedrock_live():
    from core.agent.bedrock_dialog import BedrockDialogSession
    from core.agent.tool_registry import ToolRegistry

    model_id = _configured_model_ids().get("agent_dialog")
    kwargs = {"model_id": model_id} if model_id else {}
    session = BedrockDialogSession(
        registry=ToolRegistry(), agent_type="ops", region=_region(), **kwargs,
    )
    reply = session.chat("헬스체크입니다. '정상' 한 단어로만 답하세요.")
    ok = bool(reply.strip()) and "Bedrock 호출 실패" not in reply
    return ok, f"model={model_id or '(세션 기본값)'}, 응답 {len(reply)}자"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="Bedrock converse 실호출 1회 포함")
    args = ap.parse_args()
    print("=" * 64)
    print("운영·감사 에이전트 헬스체크 (AWS)")
    print("=" * 64)
    print("\n[기본 — 빠름, LLM 토큰 소비 없음]")
    check("1. AWS 자격증명/리전", c_credentials)
    check("2. Bedrock model id 유효성", c_bedrock_models)
    check("3. 도구 스키마↔구현 정합", c_tool_contract)
    check("4. tool routing 실존성", c_tool_routing)
    check("5. case_store 백엔드", c_case_store)
    check("6. 에이전트 모듈 임포트", c_imports)
    if args.full:
        print("\n[심화 — Bedrock 실호출, 과금]")
        check("7. Bedrock converse 실호출", c_bedrock_live)
    print("=" * 64)
    n_ok = sum(results)
    print(f"결과: {n_ok}/{len(results)} 통과" + ("  ✅ 정상" if n_ok == len(results) else "  ⚠️ 확인 필요"))
    sys.exit(0 if n_ok == len(results) else 1)
