import asyncio
from bleak import BleakClient, BleakScanner

TARGET_NAME = "R1_05419"


async def main():
    print(f"{TARGET_NAME} を探しています...")
    device = await BleakScanner.find_device_by_name(TARGET_NAME, timeout=8.0)
    if not device:
        print(f"{TARGET_NAME} が見つかりませんでした。")
        return

    print(f"発見: {device.name} ({device.address}) - 接続して GATT サービスを確認します...")
    async with BleakClient(device, timeout=15.0) as client:
        print(f"接続成功: is_connected={client.is_connected}")
        for service in client.services:
            print(f"\n[Service] {service.uuid} ({service.description})")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f"  [Char] {char.uuid} | props: [{props}] | desc: {char.description}")
                if "read" in char.properties:
                    try:
                        val = await client.read_gatt_char(char.uuid)
                        print(f"         value: {val!r} (hex: {val.hex()})")
                    except Exception as e:
                        print(f"         read error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
