#!/usr/bin/env python3
"""
使用高德地图地理编码 API 补全缺失的城市信息。

策略：只对城市名为空或无法匹配的查询调用 API，最大限度减少调用次数。
输出在原始提取数据基础上增加 amap_city / amap_province / amap_district 列。

用法：
  python amap_geocode.py \
    --input "完整 qcc_extracted_v2.csv" \
    --output "完整 qcc_extracted_v3.csv" \
    --key YOUR_AMAP_KEY

高德 API 文档：https://lbs.amap.com/api/webservice/guide/api/georegeo
"""
import os, sys, argparse, json, time
import pandas as pd
import requests

API_URL = 'https://restapi.amap.com/v3/geocode/geo'
RATE_LIMIT_INTERVAL = 0.05  # 50ms between requests (高德免费版 QPS 限制约 20)


def geocode_address(address: str, key: str, city: str = '') -> dict | None:
    """调用高德地理编码 API，返回 {province, city, district, adcode, level} 或 None。"""
    params = {
        'key': key,
        'address': address,
        'output': 'JSON',
    }
    if city:
        params['city'] = city

    try:
        resp = requests.get(API_URL, params=params, timeout=10)
        data = resp.json()

        if data.get('status') == '1' and data.get('geocodes'):
            geo = data['geocodes'][0]
            return {
                'province': geo.get('province', ''),
                'city': geo.get('city', ''),
                'district': geo.get('district', ''),
                'adcode': geo.get('adcode', ''),
                'level': geo.get('level', ''),
                'location': geo.get('location', ''),
            }
        return None
    except Exception as e:
        print(f"  [WARN] API error for '{address[:30]}...': {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='Input CSV (extracted addresses)')
    ap.add_argument('--output', default='', help='Output CSV (default: overwrite)')
    ap.add_argument('--key', required=True, help='高德 Web 服务 API Key')
    ap.add_argument('--addr-col', default='Door Address\n门店营业地址',
                    help='Address column name in input CSV')
    ap.add_argument('--city-col', default='市', help='City column to check/filter')
    ap.add_argument('--only-empty', action='store_true',
                    help='Only geocode rows where city is empty (default: also include mismatched)')
    ap.add_argument('--batch-size', type=int, default=100,
                    help='Print progress every N rows')
    ap.add_argument('--dry-run', action='store_true',
                    help='Count how many rows would be geocoded without calling API')
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding='utf-8-sig')
    output = args.output or args.input

    city_col = args.city_col
    addr_col = args.addr_col

    if addr_col not in df.columns:
        print(f"ERROR: addr column '{addr_col}' not found. Available: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(1)

    # Identify rows that need geocoding
    city_values = df[city_col].fillna('').astype(str).str.strip()
    needs_geocode = city_values.isin(['', 'nan']) | (city_values == '')

    n_geocode = needs_geocode.sum()
    print(f"Total rows: {len(df)}", file=sys.stderr)
    print(f"Rows needing geocode: {n_geocode}", file=sys.stderr)

    if args.dry_run:
        print(f"\n[DRY RUN] Would geocode {n_geocode} addresses", file=sys.stderr)
        # Show samples
        samples = df[needs_geocode][addr_col].head(10)
        print("\nSample addresses:", file=sys.stderr)
        for addr in samples:
            print(f"  {str(addr)[:80]}", file=sys.stderr)
        return

    if n_geocode == 0:
        print("No rows need geocoding. Exiting.", file=sys.stderr)
        return

    # Add amap columns
    df['amap_province'] = ''
    df['amap_city'] = ''
    df['amap_district'] = ''
    df['amap_adcode'] = ''

    # Geocode
    t0 = time.time()
    success = 0
    fail = 0
    indices = df.index[needs_geocode].tolist()

    for i, idx in enumerate(indices):
        addr = str(df.at[idx, addr_col]) if pd.notna(df.at[idx, addr_col]) else ''

        if not addr.strip():
            fail += 1
            continue

        result = geocode_address(addr, args.key)
        if result:
            df.at[idx, 'amap_province'] = result.get('province', '')
            df.at[idx, 'amap_city'] = result.get('city', '')
            df.at[idx, 'amap_district'] = result.get('district', '')
            df.at[idx, 'amap_adcode'] = result.get('adcode', '')
            success += 1
        else:
            fail += 1

        if (i + 1) % args.batch_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_geocode - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_geocode}] ok={success} fail={fail} "
                  f"rate={rate:.1f}/s eta={eta:.0f}s", file=sys.stderr)

        time.sleep(RATE_LIMIT_INTERVAL)

    elapsed = time.time() - t0
    print(f"\nGeocoding done in {elapsed:.0f}s", file=sys.stderr)
    print(f"  Success: {success}/{n_geocode} ({success/n_geocode*100:.1f}%)", file=sys.stderr)
    print(f"  Failed: {fail}", file=sys.stderr)

    # Merge: use amap_city to fill empty city column
    # 高德返回的 city 对于直辖市可能为空（直辖市 province 就是城市）
    municipalities = {'北京', '上海', '天津', '重庆',
                      '北京市', '上海市', '天津市', '重庆市'}

    for idx in indices:
        orig_city = str(df.at[idx, city_col]) if pd.notna(df.at[idx, city_col]) else ''
        amap_city = str(df.at[idx, 'amap_city']) if pd.notna(df.at[idx, 'amap_city']) else ''
        amap_prov = str(df.at[idx, 'amap_province']) if pd.notna(df.at[idx, 'amap_province']) else ''

        if (not orig_city or orig_city == 'nan') and amap_city and amap_city != '[]':
            df.at[idx, city_col] = amap_city
        elif (not orig_city or orig_city == 'nan') and not amap_city:
            # 直辖市：高德 city 返回空，用 province
            if any(m in amap_prov for m in municipalities):
                df.at[idx, city_col] = amap_prov

    df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output}", file=sys.stderr)

    # Summary
    filled = sum(1 for idx in indices
                 if (not str(df.at[idx, city_col]) or str(df.at[idx, city_col]) == 'nan')
                 and df.at[idx, 'amap_city'])
    print(f"City column filled by amap: {success}", file=sys.stderr)


if __name__ == '__main__':
    main()
