"""빌드된 React 진입점과 정적 자산을 실제 FastAPI 라우트에서 검증한다."""
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from webapp.server import create_app

with tempfile.TemporaryDirectory() as directory:
    with TestClient(create_app(directory), base_url="http://127.0.0.1") as client:
        response = client.get("/")
        assert response.status_code == 200, response.text
        assert 'id="root"' in response.text
        urls = re.findall(r'(?:src|href)="(/assets/[^"\s]+)"', response.text)
        assert len(urls) >= 2, "React JS와 CSS 빌드 필요"
        for url in urls:
            asset = client.get(url)
            assert asset.status_code == 200, url
            assert len(asset.content) > 100, url
        state = client.get("/api/state")
        assert state.status_code == 200, state.text
        assert state.json()["active_job"] is None
        assert client.post("/api/jobs/train").status_code == 403
        print(f"React entry, {len(urls)} assets and same-origin API routes passed")
