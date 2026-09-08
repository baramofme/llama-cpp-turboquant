# SSE ping 유지보수 (라우터 중간 드랍 방지)

> 배경: llm-main 라우터가 긴 prefill 동안 중간 게이트웨이/클라이언트 연결이 끊겨
> "http client error: Connection handling canceled" 후 요청이 취소되는 현상의 해결책.
> 실제 사례/로그 분석은 `session-overview/SESSION-2026-09-06-MASTER.md` 및 구현 계획 참고.

## 현상

라우터(router mode)는 업스트림 클라이언트에 SSE를 스트리밍한다. 두 요청이 한 자식
인스턴스에 겹치면 prefill이 수십 초 걸리고 그 사이 SSE가 잠시 쉬면, 중간 프록시/게이트웨이
(read/idle timeout)가 연결을 끊는다. 끊김은 소스상 정확히 아래처럼 전파됨:

```
업스트림 클라 연결 종료 -> 라우터 프록시 pipe broken -> httplib Error::Canceled
 -> 라우터가 자식 소켓 닫음 -> 자식이 해당 task cancel ("cancel task")
```

## 해결책 (config.ini 전역)

`/opt/llm/llama-cpp/main-llm-config.ini` 의 `[*]` 전역 섹션에 추가:

```ini
[*]
# 긴 prefill 동안 중간 게이트웨이/라우터가 연결을 끊지 않도록 SSE ping을 자주 보냄
sse-ping-interval = 10
# 라우터 read/write timeout (기본 3600s, 큰 컨텍스트 prefill 대비 유지)
timeout = 3600
```

적용 방법: config.ini는 llama-server 시작 시 한 번 읽으므로 **컨테이너 재기동 필요**
(`/tmp/llm-main-restart.sh` 또는 `docker restart llm-main`).

## 관련 소스 (이식 feature 대비 맵핑)

| 파일 | 역할 |
|---|---|
| `tools/server/server-models.cpp:2495-2497` | 라우터 프록시 `http client error` 로그 |
| `tools/server/server-models.cpp:2348-2523` | `server_http_proxy` (파이프 릴레이) |
| `tools/server/server-common.h:549-611` | `server_pipe` (read/write, broken pipe) |
| `tools/server/server-http.cpp:597` | `req.should_stop = is_connection_closed` (피어 disconnect) |
| `vendor/cpp-httplib/httplib.cpp:9018-9022` | `is_connection_closed` = `!is_socket_alive(sock)` |
| `vendor/cpp-httplib/httplib.cpp:5989` | `Error::Canceled` = "Connection handling canceled" |
| `tools/server/server-context.cpp:507-511` | 자식 slot `release()` / "stop processing" |

## 핵심 교훈

1. `--sse-ping-interval`은 **라우터 레벨** 파라미터 → `[*]` 전역에 넣어야 자식에도 상속.
2. 기본값은 30초. 10초 정도로 줄이면 긴 prefill 중 게이트웨이 idle timeout을 실효적으로 막음.
3. llm-main **앞단** 별도 게이트웨이(n8n/traefik/open-webui 등)가 있으면 그쪽 응답/read
   타임아웃도 prefill 시간보다 길게 잡아야 완전 해소.
4. 소스 분석상 이는 **버그가 아니라** 피어 연결 종료의 의도된 전파. 방어는 ping/타임아웃 튜닝.
