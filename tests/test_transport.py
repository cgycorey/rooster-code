import asyncio

import httpx

from rooster_code.transport_legacy import RawAnthropicHTTPClient, TransportHTTPError


def test_raw_http_client_maps_timeout_to_api_error() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = RawAnthropicHTTPClient(
            api_key="test-key",
            base_url="https://custom.example/api",
            http_client=http_client,
        )

        try:
            await client.messages.create(
                model="glm-5:cloud",
                max_tokens=32,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            )
        except TransportHTTPError as exc:
            assert exc.status_code == 504
            assert "timed out" in str(exc)
        else:
            raise AssertionError("expected TransportHTTPError")
        finally:
            await client.close()

    asyncio.run(run_test())
