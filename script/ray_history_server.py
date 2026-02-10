#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import html
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import requests


NAMESPACE = os.getenv("K8S_NAMESPACE", "kaiyuan-ray")
PORT = int(os.getenv("HISTORY_SERVER_PORT", "8080"))
API_SERVER = os.getenv("K8S_API_SERVER", "https://kubernetes.default.svc")
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _read_service_account_token() -> str:
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def _fetch_rayjobs() -> list[dict[str, Any]]:
    token = _read_service_account_token()
    url = f"{API_SERVER}/apis/ray.io/v1/namespaces/{NAMESPACE}/rayjobs"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        verify=CA_PATH,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def _fmt_time(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return value


def _render_html(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in sorted(
        items,
        key=lambda x: x.get("metadata", {}).get("creationTimestamp", ""),
        reverse=True,
    ):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        spec = item.get("spec", {})

        name = meta.get("name", "-")
        created = _fmt_time(meta.get("creationTimestamp"))
        job_status = status.get("jobStatus", "-")
        deploy_status = status.get("jobDeploymentStatus", "-")
        cluster_name = status.get("rayClusterName", "-")
        start_time = _fmt_time(status.get("startTime"))
        end_time = _fmt_time(status.get("endTime"))
        message = status.get("message", "")
        entrypoint = spec.get("entrypoint", "")

        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(job_status)}</td>"
            f"<td>{html.escape(deploy_status)}</td>"
            f"<td>{html.escape(cluster_name)}</td>"
            f"<td>{html.escape(created)}</td>"
            f"<td>{html.escape(start_time)}</td>"
            f"<td>{html.escape(end_time)}</td>"
            f"<td><code>{html.escape(entrypoint)}</code></td>"
            f"<td>{html.escape(message)}</td>"
            "</tr>"
        )

    body_rows = "\n".join(rows) if rows else '<tr><td colspan="9">No RayJobs found.</td></tr>'

    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Ray Job History</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    .sub {{ margin-bottom: 16px; color: #6b7280; }}
    .actions {{ margin-bottom: 16px; }}
    button {{ border: 1px solid #d1d5db; background: white; border-radius: 6px; padding: 6px 10px; cursor: pointer; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>Ray Job History</h1>
  <div class="sub">Namespace: <b>{html.escape(NAMESPACE)}</b> | Last refresh: {html.escape(now)}</div>
  <div class="actions">
    <button onclick="location.reload()">Refresh</button>
  </div>
  <table>
    <thead>
      <tr>
        <th>RayJob</th>
        <th>Job Status</th>
        <th>Deployment Status</th>
        <th>Ray Cluster</th>
        <th>Created</th>
        <th>Started</th>
        <th>Ended</th>
        <th>Entrypoint</th>
        <th>Message</th>
      </tr>
    </thead>
    <tbody>
      {body_rows}
    </tbody>
  </table>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == "/api/jobs":
                items = _fetch_rayjobs()
                payload = json.dumps({"items": items}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            items = _fetch_rayjobs()
            content = _render_html(items).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            msg = f"Ray history server error: {exc}".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Ray history server listening on 0.0.0.0:{PORT} (namespace={NAMESPACE})")
    server.serve_forever()


if __name__ == "__main__":
    main()
