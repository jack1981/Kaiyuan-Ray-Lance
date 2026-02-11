#!/usr/bin/env python3
"""Simple HTTP server that renders RayJob history from Kubernetes API.

The server is intended for in-cluster use with service-account credentials and
provides both HTML and JSON (`/api/jobs`) responses.
"""

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
    """Read Kubernetes service-account bearer token from mounted secret.

    Inputs/outputs:
        No inputs; returns token string.

    Side effects:
        Reads token file from filesystem.

    Assumptions:
        Script runs in Kubernetes pod with standard service-account mount paths.
    """
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def _fetch_rayjobs() -> list[dict[str, Any]]:
    """Fetch RayJob resources from Kubernetes API for configured namespace.

    Inputs/outputs:
        No inputs; returns list of RayJob objects (dict form).

    Side effects:
        Performs HTTPS request to Kubernetes API server.

    Assumptions:
        Service-account RBAC allows listing RayJob resources.
    """
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
    """Format ISO-like timestamps for table display.

    Args:
        value: Raw timestamp string or None.

    Returns:
        Human-readable UTC timestamp string or fallback `-`/raw value.

    Side effects:
        None.

    Assumptions:
        Kubernetes timestamps are ISO8601 with trailing `Z`.
    """
    if not value:
        return "-"
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return value


def _render_html(items: list[dict[str, Any]]) -> str:
    """Render RayJob list as standalone HTML table page.

    Args:
        items: RayJob resource objects from Kubernetes API.

    Returns:
        HTML document string.

    Side effects:
        None.

    Assumptions:
        RayJob objects follow standard metadata/spec/status structure.
    """
    rows = []
    # NOTE(readability): Newest-first ordering mirrors operational expectations
    # when tracking latest submitted RayJob executions.
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
        """Serve JSON API or HTML history page for incoming GET requests.

        Inputs/outputs:
            Handles request path and writes HTTP response.

        Side effects:
            Performs Kubernetes API calls and socket writes.

        Assumptions:
            Any unexpected exception should return HTTP 500 with plain-text error.
        """
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
        """Suppress default noisy request logging.

        Inputs/outputs:
            Accepts standard BaseHTTPRequestHandler log args and returns None.

        Side effects:
            None (intentional no-op).

        Assumptions:
            Higher-level container logs capture enough operational signal.
        """
        return


def main() -> None:
    """Start HTTP server and block forever.

    Inputs/outputs:
        No inputs; binds configured host/port and serves requests.

    Side effects:
        Opens listening socket and blocks process lifetime.

    Assumptions:
        Binding on `0.0.0.0` is desired for in-cluster service access.
    """
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Ray history server listening on 0.0.0.0:{PORT} (namespace={NAMESPACE})")
    server.serve_forever()


if __name__ == "__main__":
    main()
