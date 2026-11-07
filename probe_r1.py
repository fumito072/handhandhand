import asyncio
from bleak import BleakClient, BleakScanner
from Crypto.Cipher import AES

TARGET_NAME = "R1_05419"
NOTIFY_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
WRITE_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"

AES_KEY = bytes.fromhex("df98b715d5c6ed2b25817b6f2554124a")
AES_IV = bytes.fromhex("2841ae97419c2973296a0d4bdfe19a4f")
V3_MAGIC = b"\x00UT25"


def ble_encrypt(data: bytes) -> bytes:
    return AES.new(AES_KEY, AES.MODE_CFB, iv=AES_IV, segment_size=128).encrypt(data)


def ble_decrypt(data: bytes) -> bytes:
    return AES.new(AES_KEY, AES.MODE_CFB, iv=AES_IV, segment_size=128).decrypt(data)


def build_simple(instruction: int, data: bytes = b"") -> bytes:
    payload = bytes([0x52, len(data) + 4, instruction]) + data
    checksum = (-sum(payload)) & 0xFF
    return ble_encrypt(payload + bytes([checksum]))


def build_chunked(instruction: int, chunk_data: bytes, idx: int = 1, total: int = 1) -> bytes:
    payload = bytes([0x52, len(chunk_data) + 6, instruction, idx, total]) + chunk_data
    checksum = (-sum(payload)) & 0xFF
    return ble_encrypt(payload + bytes([checksum]))


async def main():
    device = await BleakScanner.find_device_by_name(TARGET_NAME, timeout=8.0)
    if not device:
        print(f"{TARGET_NAME} が見つかりませんでした。")
        return

    print(f"接続中: {device.name} ({device.address})")
    rx_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def on_notify(sender, data: bytearray):
        raw = bytes(data)
        rx_queue.put_nowait(raw)

    async with BleakClient(device, timeout=15.0) as client:
        print(f"接続完了 (MTU={client.mtu_size})")
        await client.start_notify(NOTIFY_UUID, on_notify)

        # 1. Send HANDSHAKE ("unitree")
        print("\n[1] HANDSHAKE を送信中...")
        await client.write_gatt_char(
            WRITE_UUID, build_chunked(0x01, b"unitree", idx=1, total=1), response=True
        )

        is_v3 = False
        try:
            raw = await asyncio.wait_for(rx_queue.get(), timeout=5.0)
            if raw.startswith(V3_MAGIC):
                is_v3 = True
                print(f"-> V3 プロトコル応答を受信! raw={raw.hex()} | ascii={raw!r}")
                if len(raw) >= 13 and raw[5] == 0xF1:
                    ver = raw[6]
                    flags = raw[7]
                    sn_len = raw[12]
                    sn = raw[13 : 13 + sn_len].decode("utf-8", errors="replace")
                    print(f"   V3 Version={ver}, flags={flags}, SN={sn!r}")
            else:
                plain = ble_decrypt(raw)
                print(f"-> V1 AES-CFB 応答を受信! decrypted={plain.hex()}")
                if len(plain) >= 4 and plain[0] == 0x51:
                    print(f"   cmd=0x{plain[2]:02x}, status=0x{plain[3]:02x}")
        except asyncio.TimeoutError:
            print("-> HANDSHAKE 応答タイムアウト")

        if not is_v3:
            # Try GET_SN (0x02) and GET_AP_MAC (0x07) via V1
            print("\n[2] V1 GET_SN (0x02) を送信中...")
            await client.write_gatt_char(WRITE_UUID, build_simple(0x02), response=True)
            sn_chunks = {}
            try:
                while True:
                    raw = await asyncio.wait_for(rx_queue.get(), timeout=2.0)
                    plain = ble_decrypt(raw)
                    print(f"   GET_SN chunk: {plain.hex()}")
                    if len(plain) >= 6 and plain[0] == 0x51 and plain[2] == 0x02:
                        idx, total = plain[3], plain[4]
                        sn_chunks[idx] = plain[5 : plain[1] - 1]
                        if len(sn_chunks) >= total:
                            sn = b"".join(sn_chunks[i] for i in sorted(sn_chunks)).decode(
                                "utf-8", errors="replace"
                            ).rstrip("\x00")
                            print(f"-> シリアル番号 (SN): {sn}")
                            break
            except asyncio.TimeoutError:
                print("   (GET_SN 終了/タイムアウト)")

            print("\n[3] V1 GET_AP_MAC (0x07) を送信中...")
            await client.write_gatt_char(WRITE_UUID, build_simple(0x07), response=True)
            try:
                raw = await asyncio.wait_for(rx_queue.get(), timeout=3.0)
                plain = ble_decrypt(raw)
                print(f"   GET_AP_MAC plain: {plain.hex()}")
                if len(plain) >= 5 and plain[0] == 0x51 and plain[2] == 0x07:
                    mac_bytes = plain[3 : plain[1] - 1]
                    try:
                        mac_str = mac_bytes.decode("utf-8").strip("\x00")
                    except Exception:
                        mac_str = ":".join(f"{b:02x}" for b in mac_bytes)
                    print(f"-> AP MAC: {mac_str}")
            except asyncio.TimeoutError:
                print("   (GET_AP_MAC タイムアウト)")


if __name__ == "__main__":
    asyncio.run(main())
