# -*- coding: utf-8 -*-
"""
[1번 담당 검증] 좌표 캐시 재사용 확인

geo_lookup.lookup_region()은 캐시 키에 반경을 붙여 저장한다.
격자 슬라이더를 움직여 반경이 바뀌면 키가 달라져 캐시가 안 먹고
Nominatim을 다시 호출하던 문제가 있었다.

수정 후에는 같은 지역의 다른 반경 항목에서 중심점을 꺼내
새 반경으로 다시 계산하므로 API 재호출이 없다.

이 스크립트는 Nominatim을 응답 없는 로컬 서버로 바꿔치기해서
"인터넷이 안 되는 상태"를 만든 뒤, 캐시만으로 좌표가 나오는지 확인한다.
0.0초로 성공하면 API를 안 탄 것이고, 10초 뒤 실패하면 API를 탄 것이다.

실행:
    python tools/verify_region_cache.py
"""
import json, os, sys, shutil, socket, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import geo_lookup

# 인터넷 없는 상황을 확실히 만들기 위해 응답 없는 서버로 돌림
srv=socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
srv.bind(("127.0.0.1",0)); srv.listen(8); port=srv.getsockname()[1]
threading.Thread(target=lambda:[srv.accept() for _ in iter(int,1)], daemon=True).start()
geo_lookup.NOMINATIM_URL=f"http://127.0.0.1:{port}/search"

# 실제 캐시 파일 복사본 사용
import tempfile
_tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(ROOT,'region_cache.json'), os.path.join(_tmp,'cache_test.json'))
geo_lookup.CACHE_PATH=os.path.join(_tmp,'cache_test.json')

print("저장된 키:", list(json.load(open(geo_lookup.CACHE_PATH,encoding='utf-8')).keys()))
print()
for gl in (100, 150, 200, 175):
    margin = 10*gl/2
    t=time.time()
    try:
        r=geo_lookup.lookup_region("강남역", margin_m=margin)
        ok=f"성공  lat {r['lat_min']}~{r['lat_max']}"
    except Exception as e:
        ok=f"실패  {type(e).__name__}"
    print(f"grid_length={gl:>4} (반경 {margin:>6}m)  {time.time()-t:5.1f}초  {ok}")
print()
print("홍대입구 (반경 없는 옛날 키만 있던 지역)")
t=time.time()
try:
    r=geo_lookup.lookup_region("홍대입구", margin_m=500)
    print(f"  {time.time()-t:.1f}초  성공  lat {r['lat_min']}~{r['lat_max']}")
except Exception as e:
    print(f"  {time.time()-t:.1f}초  실패  {type(e).__name__}")
print()
print("캐시에 없는 지역 (여전히 실패해야 정상)")
t=time.time()
try:
    geo_lookup.lookup_region("부산 해운대", margin_m=500)
    print("  성공 ← 잘못됨")
except Exception as e:
    print(f"  {time.time()-t:.1f}초  실패  {type(e).__name__}  ← 정상")
