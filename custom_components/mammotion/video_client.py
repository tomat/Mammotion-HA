"""Video-only Mammotion cloud client."""

from __future__ import annotations

import asyncio
import time

from aiohttp import ClientSession
from pymammotion.client import MammotionClient
from pymammotion.http.http import MammotionHTTP
from pymammotion.http.model.camera_stream import StreamSubscriptionResponse
from pymammotion.http.model.http import Response
from pymammotion.transport.base import LoginFailedError, TransportType
from pymammotion.utility.device_type import DeviceType

from .const import LOGGER


class MammotionVideoStreamClient:
    """Small Mammotion HTTP client used only for camera stream tokens.

    This deliberately avoids ``MammotionClient.login_and_initiate_cloud()`` so a
    secondary camera account does not register devices, open MQTT transports, or
    start background watchers.
    """

    def __init__(
        self,
        account: str,
        password: str,
        session: ClientSession,
        ha_version: str,
    ) -> None:
        """Initialize the video-only client."""
        self.account = account
        self._password = password
        self._http = MammotionHTTP(
            account=account,
            password=password,
            session=session,
            ha_version=ha_version,
        )
        self.user_account: str | None = None
        self._logged_in = False
        self._session = session
        self._ha_version = ha_version
        self._command_client: MammotionClient | None = None
        self._command_lock = asyncio.Lock()

    async def async_login(self) -> None:
        """Log in to Mammotion HTTP without opening MQTT transports."""
        response = await self._http.login_v2(self.account, self._password)
        if response.code != 0 or response.data is None:
            raise LoginFailedError(self.account, response.msg or "login failed")

        self.user_account = response.data.userInformation.userAccount
        self._logged_in = True

    async def async_get_stream_subscription(
        self, device_name: str, iot_id: str
    ) -> Response[StreamSubscriptionResponse] | None:
        """Fetch an Agora stream token through the secondary account."""
        if not self._logged_in:
            await self.async_login()

        return await self._http.get_stream_subscription(
            iot_id,
            DeviceType.is_yuka(device_name),
        )

    async def async_send_video_command(
        self,
        device_name: str,
        iot_id: str,
        command: str,
        **kwargs,
    ) -> bool:
        """Send a mower video command through the secondary account."""
        bypass_local_rate_limit = bool(
            kwargs.pop("_bypass_local_rate_limit", False)
        )
        quiet_rate_limit_clear = bool(kwargs.pop("_quiet_rate_limit_clear", False))
        record_cmd = bool(kwargs.pop("_record_cmd", False))
        client, command_device_name = await self._async_get_command_client(
            device_name, iot_id
        )
        if bypass_local_rate_limit:
            self._clear_command_client_rate_limits(
                client,
                command_device_name,
                reason=f"command {command}",
                quiet=quiet_rate_limit_clear,
            )
        await client.send_command_with_args(
            command_device_name,
            command,
            prefer_ble=False,
            _record_cmd=record_cmd,
            **kwargs,
        )
        return True

    async def _async_get_command_client(
        self, device_name: str, iot_id: str
    ) -> tuple[MammotionClient, str]:
        """Return a secondary-account command client with polling disabled."""
        async with self._command_lock:
            if self._command_client is None:
                self._command_client = MammotionClient(ha_version=self._ha_version)
                await self._command_client.login_and_initiate_cloud(
                    self.account,
                    self._password,
                    self._session,
                )
                await self._disable_command_client_polling(self._command_client)

            command_device_name = self._resolve_command_device_name(
                self._command_client, device_name, iot_id
            )
            if command_device_name is None:
                await self._command_client.stop()
                self._command_client = None
                raise RuntimeError(
                    "secondary Mammotion video account cannot access "
                    f"{device_name} ({iot_id})"
                )

            return self._command_client, command_device_name

    @staticmethod
    async def _disable_command_client_polling(client: MammotionClient) -> None:
        """Keep secondary MQTT available for commands without background polls."""
        for handle in list(client._device_registry.all_devices):  # noqa: SLF001
            handle.set_mow_path_fetch_enabled(value=False)
            handle.set_full_map_fetch_enabled(value=False)
            await handle.stop_polling()

    @staticmethod
    def _resolve_command_device_name(
        client: MammotionClient, device_name: str, iot_id: str
    ) -> str | None:
        """Resolve the primary device identity in the secondary account."""
        if client.mower(device_name) is not None:
            return device_name

        mapped_name = client._iot_id_to_device_id.get(iot_id)  # noqa: SLF001
        if mapped_name and client.mower(mapped_name) is not None:
            return mapped_name

        return None

    @staticmethod
    def _clear_command_client_rate_limits(
        client: MammotionClient,
        device_name: str,
        *,
        reason: str,
        quiet: bool = False,
    ) -> None:
        """Clear local secondary-account send-budget latches before nudges."""
        handle = client.mower(device_name)
        if handle is None:
            return

        now = time.monotonic()
        for transport_type in (
            TransportType.CLOUD_ALIYUN,
            TransportType.CLOUD_MAMMOTION,
        ):
            transport = handle.get_transport(transport_type)
            if transport is None:
                continue

            send_timestamps = getattr(transport, "_send_timestamps", None)
            send_count = len(send_timestamps) if send_timestamps is not None else 0
            rate_limited = bool(getattr(transport, "is_rate_limited", False))

            gateway = getattr(transport, "_cloud_gateway", None)
            gateway_until = getattr(gateway, "_rate_limited_until", 0.0)
            gateway_limited = (
                isinstance(gateway_until, (int, float)) and gateway_until > now
            )

            if (rate_limited or gateway_limited or send_count) and not quiet:
                LOGGER.warning(
                    "Clearing secondary-account %s send budget for %s on %s "
                    "(transport_limited=%s, gateway_limited=%s, sent=%d)",
                    transport_type.value,
                    reason,
                    device_name,
                    rate_limited,
                    gateway_limited,
                    send_count,
                )

            if hasattr(transport, "_rate_limited_until"):
                transport._rate_limited_until = 0.0  # noqa: SLF001
            if send_timestamps is not None:
                send_timestamps.clear()
            if gateway is not None and hasattr(gateway, "_rate_limited_until"):
                gateway._rate_limited_until = 0.0  # noqa: SLF001
            if gateway is not None and hasattr(gateway, "_rate_limit_backoff"):
                gateway._rate_limit_backoff = 60.0  # noqa: SLF001

    async def async_stop(self) -> None:
        """Log out the secondary video account."""
        if self._command_client is not None:
            await self._command_client.stop()
            self._command_client = None

        if self._http.login_info is None:
            return

        try:
            await self._http.logout()
        except Exception:
            LOGGER.debug(
                "Failed to log out Mammotion video account %s",
                self.account,
                exc_info=True,
            )
