#!/usr/bin/env python3
"""
地址省市区提取 — 正则预提取 + cpca 兜底

管线：
  Stage 0: 预处理（空地址跳过、strip、全角统一）
  Stage 1: 正则提取省/市/区（确定性 100%）
  Stage 2: cpca 兜底（补全 Stage 1 缺失字段）
  Stage 3: 后处理校验（直辖市修正、置信度计算）

用法：
  python extract_address.py --input 完整\ qcc.csv --output result.csv --encoding gb18030
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── cpca 延迟加载 ──────────────────────────────────────────────
_cpca_module = None


def _get_cpca():
    global _cpca_module
    if _cpca_module is None:
        import cpca as _c
        _cpca_module = _c
    return _cpca_module


# ── 正则模式 ──────────────────────────────────────────────────
PROVINCE_RE = re.compile(r'([\u4e00-\u9fa5]{2,8}(?:省|自治区|特别行政区))')
CITY_RE = re.compile(r'([\u4e00-\u9fa5]{2,6}(?:市|地区|自治州))')
DISTRICT_RE = re.compile(r'([\u4e00-\u9fa5]{2,6}(?:区|县|旗|自治县|自治旗|新区|开发区))')

MUNICIPALITIES = {'北京市', '上海市', '天津市', '重庆市'}
DISTRICTLESS_CITIES = {'东莞市', '中山市', '儋州市', '嘉峪关市'}

# 全角数字 → 半角
_FWC = str.maketrans('０１２３４５６７８９', '0123456789')


def _normalize(addr: str) -> str:
    """Stage 0: 预处理"""
    if not isinstance(addr, str):
        return ''
    addr = addr.strip().translate(_FWC)
    addr = re.sub(r'\s+', ' ', addr)
    return addr


def _clean_extracted(name: str, level: str, known_city: str = '') -> str:
    """清理正则提取结果：去除跨级匹配的多余前缀。
    例如 '省台州市路桥区' → '路桥区'，'浙江省温州市' → '温州市'
    known_city: 已提取的市名（不带'市'后缀），用于清理区名前缀
    """
    if not name:
        return name
    if level == 'city':
        # 去除省级后缀（省、自治区）及其之前的所有内容
        for marker in ('省', '自治区', '特别行政区'):
            idx = name.rfind(marker)
            if idx >= 0:
                name = name[idx + len(marker):]
    elif level == 'dist':
        # 去除市级后缀（市、地区、自治州）及其之前的所有内容
        for marker in ('市', '地区', '自治州'):
            idx = name.rfind(marker)
            if idx >= 0:
                name = name[idx + len(marker):]
        # 同时去除省级后缀
        for marker in ('省', '自治区', '特别行政区'):
            idx = name.rfind(marker)
            if idx >= 0:
                name = name[idx + len(marker):]
        # 去除已知市名前缀（如'石家庄长安区' → '长安区'）
        if known_city and known_city.endswith('市'):
            city_core = known_city[:-1]  # 去掉末尾'市'
            if city_core and name.startswith(city_core):
                name = name[len(city_core):]
    return name


def extract_by_regex(addr: str) -> dict:
    """Stage 1: 正则提取，返回 {prov, city, dist, prov_src, city_src, dist_src}"""
    result = {
        'prov': '', 'city': '', 'dist': '',
        'prov_src': 'none', 'city_src': 'none', 'dist_src': 'none',
    }

    # 直辖市优先检测
    for m in MUNICIPALITIES:
        if m in addr:
            result['prov'] = m
            result['prov_src'] = 'regex'
            result['city'] = m
            result['city_src'] = 'regex'
            break

    # 省级
    if not result['prov']:
        m = PROVINCE_RE.search(addr)
        if m:
            result['prov'] = m.group(1)
            result['prov_src'] = 'regex'

    # 市级（非直辖市）
    if not result['city']:
        m = CITY_RE.search(addr)
        if m:
            raw = m.group(1)
            result['city'] = _clean_extracted(raw, 'city')
            result['city_src'] = 'regex'

    # 区县级
    m = DISTRICT_RE.search(addr)
    if m:
        raw = m.group(1)
        result['dist'] = _clean_extracted(raw, 'dist', result['city'])
        result['dist_src'] = 'regex'

    return result


def extract_by_cpca(addr: str) -> dict:
    """Stage 2: cpca 提取"""
    c = _get_cpca()
    df = c.transform([addr])
    row = df.iloc[0]
    prov = row['省'] if row['省'] and row['省'] != '' else ''
    city = row['市'] if row['市'] and row['市'] != '' else ''
    dist = row['区'] if row['区'] and row['区'] != '' else ''

    # 直辖市修正：cpca 把直辖市放省列，市列显示"市辖区"
    if prov in MUNICIPALITIES and city == '市辖区':
        city = prov

    return {'prov': prov, 'city': city, 'dist': dist}


def merge(regex_r: dict, cpca_r: dict) -> dict:
    """合并结果：正则优先，cpca 只补空字段"""
    final = {}
    for field in ('prov', 'city', 'dist'):
        src_field = f'{field}_src'
        if regex_r[field]:
            final[field] = regex_r[field]
            final[src_field] = regex_r[src_field]
        elif cpca_r.get(field, ''):
            final[field] = cpca_r[field]
            final[src_field] = 'cpca'
        else:
            final[field] = ''
            final[src_field] = 'none'

    # 后处理：用合并后的市名清理区名前缀（如'石家庄长安区' → '长安区'）
    if final['city'] and final['dist'] and final['dist_src'] == 'regex':
        city_core = final['city'].rstrip('市')
        if city_core and final['dist'].startswith(city_core):
            final['dist'] = final['dist'][len(city_core):]

    # 不设区的地级市：区=空是正常的
    if final['city'] in DISTRICTLESS_CITIES and not final['dist']:
        pass  # 保留空，这是正确的

    # 置信度
    sources = [final['prov_src'], final['city_src'], final['dist_src']]
    if all(s == 'none' for s in sources):
        final['confidence'] = 'none'
    elif any(s == 'regex' for s in sources) and all(s in ('regex', 'none') for s in sources):
        if all(s == 'none' for s in sources):
            final['confidence'] = 'none'
        else:
            final['confidence'] = 'high'
    elif any(s == 'regex' for s in sources):
        final['confidence'] = 'mid'
    elif any(s == 'cpca' for s in sources):
        final['confidence'] = 'mid'
    else:
        final['confidence'] = 'low'

    return final


def process_address(addr: str) -> dict:
    """处理单条地址，返回完整结果"""
    # Stage 0
    addr = _normalize(addr)
    if not addr:
        return {
            '省': '', '市': '', '区': '',
            '置信度': 'none',
            '省来源': 'none', '市来源': 'none', '区来源': 'none',
        }

    # Stage 1: 正则
    regex_r = extract_by_regex(addr)

    # Stage 2: cpca 兜底（只有正则有缺失字段时才调用）
    has_gap = not regex_r['prov'] or not regex_r['city'] or not regex_r['dist']
    cpca_r = extract_by_cpca(addr) if has_gap else {'prov': '', 'city': '', 'dist': ''}

    # Stage 3: 合并
    final = merge(regex_r, cpca_r)

    return {
        '省': final['prov'],
        '市': final['city'],
        '区': final['dist'],
        '置信度': final['confidence'],
        '省来源': final['prov_src'],
        '市来源': final['city_src'],
        '区来源': final['dist_src'],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='输入 CSV/Excel 文件')
    ap.add_argument('--output', default='', help='输出文件（默认: input_extracted.csv）')
    ap.add_argument('--encoding', default='gb18030', help='输入文件编码（默认 gb18030）')
    ap.add_argument('--addr-col', default='', help='地址列名（默认自动检测）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 条（调试用）')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not args.output:
        out_path = in_path.stem + '_extracted.csv'
    else:
        out_path = args.output

    # 加载数据
    print(f'[1/4] 加载 {in_path} ...', file=sys.stderr)
    if in_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path, encoding=args.encoding, encoding_errors='replace')

    if args.limit > 0:
        df = df.head(args.limit)

    print(f'  {len(df)} 条记录', file=sys.stderr)

    # 检测地址列
    addr_col = args.addr_col
    if not addr_col:
        for col in df.columns:
            cl = col.lower()
            if '地址' in cl or 'addr' in cl:
                addr_col = col
                break
        if not addr_col:
            print('ERROR: 找不到地址列，请用 --addr-col 指定', file=sys.stderr)
            sys.exit(1)

    print(f'  地址列: {addr_col}', file=sys.stderr)

    # 处理
    print(f'[2/4] 提取省市区 ...', file=sys.stderr)
    t0 = time.time()
    results = df[addr_col].apply(process_address)
    elapsed = time.time() - t0

    result_df = pd.DataFrame(results.tolist())

    # 合并到原始数据
    print(f'[3/4] 合并结果 ...', file=sys.stderr)
    out_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    # 统计
    print(f'[4/4] 保存到 {out_path} ...', file=sys.stderr)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    # 统计报告
    n = len(out_df)
    n_prov = (out_df['省'] != '').sum()
    n_city = (out_df['市'] != '').sum()
    n_dist = (out_df['区'] != '').sum()
    n_none = (out_df['置信度'] == 'none').sum()
    n_high = (out_df['置信度'] == 'high').sum()
    n_mid = (out_df['置信度'] == 'mid').sum()

    print(f'\n===== 提取结果 =====', file=sys.stderr)
    print(f'  总量: {n}', file=sys.stderr)
    print(f'  省: {n_prov} ({n_prov/n*100:.1f}%)', file=sys.stderr)
    print(f'  市: {n_city} ({n_city/n*100:.1f}%)', file=sys.stderr)
    print(f'  区: {n_dist} ({n_dist/n*100:.1f}%)', file=sys.stderr)
    print(f'  置信度: high={n_high} mid={n_mid} none={n_none}', file=sys.stderr)
    print(f'  耗时: {elapsed:.1f}s', file=sys.stderr)
    print(f'  输出: {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
