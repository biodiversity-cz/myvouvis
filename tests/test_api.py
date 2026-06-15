from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import app
from pipeline.types import OutputMode


def test_transcribe_routes_call_process_sheet_with_mode() -> None:
    mock_result = MagicMock()
    mock_result.as_score.return_value = {"ok": True}
    client = TestClient(app)

    routes = [
        ("/v1/transcribe-full", OutputMode.full),
        ("/v1/transcribe-dwc", OutputMode.dwc),
        ("/v1/transcribe-bbox", OutputMode.bbox),
    ]

    for path, mode in routes:
        with patch("api.app.process_sheet", return_value=mock_result) as process:
            response = client.post(
                path,
                files={"file": ("sheet.png", b"fake", "image/png")},
            )
            assert response.status_code == 200, path
            process.assert_called_once()
            assert process.call_args.kwargs["output_mode"] == mode
            mock_result.as_score.assert_called_with(mode)
            mock_result.reset_mock()
