import asyncio
from bleak import BleakScanner


async def main():
    print("BLE デバイスを 8 秒間スキャンしています...")
    devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
    print(f"\n合計 {len(devices)} 件の BLE デバイスを検出しました。\n")

    # 名前があるもの・RSSIが強い順にソート
    sorted_items = sorted(
        devices.values(),
        key=lambda item: (
            0 if (item[0].name or item[1].local_name) else 1,
            -(item[1].rssi if item[1].rssi is not None else -999),
        ),
    )

    print("=== 名前付き / 注目デバイス ===")
    found_named = 0
    for device, adv in sorted_items:
        name = device.name or adv.local_name
        uuids = adv.service_uuids or []
        mfg = adv.manufacturer_data or {}
        # 名前がある、または Unitree 系の UUID (ffe0/ffe1/ffe2 等) を持つものを表示
        has_ffe = any("ffe" in u.lower() for u in uuids)
        if name or has_ffe:
            found_named += 1
            print(f"- Name: {name!r}")
            print(f"  Address/UUID: {device.address}")
            print(f"  RSSI: {adv.rssi} dBm")
            if uuids:
                print(f"  Service UUIDs: {uuids}")
            if mfg:
                mfg_hex = {k: v.hex() for k, v in mfg.items()}
                print(f"  Manufacturer Data: {mfg_hex}")
            print()

    if found_named == 0:
        print("(名前付きの BLE デバイスは見つかりませんでした。上位 10 件の無名デバイスを表示します)")
        for device, adv in sorted_items[:10]:
            print(f"- Address: {device.address} | RSSI: {adv.rssi} dBm | UUIDs: {adv.service_uuids}")


if __name__ == "__main__":
    asyncio.run(main())
