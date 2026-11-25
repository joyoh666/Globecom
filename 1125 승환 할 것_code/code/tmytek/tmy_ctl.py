# -*- coding: utf-8 -*-
"""
[GC 1117 · FR3 API 1118] TMYTEK 장비 제어 얇은 래퍼 (UD Box 0630 = UDB · BBox Lite FR3).

TLKCore 가 없거나 --dry-run 이면 같은 호출을 화면에만 찍는다 → 장비 없이 빔 추적·대시보드를 시험할 수 있다.

BBox 쪽은 TMYTEK 이 11-18 에 준 FR3 문서를 따른다 (../3_TMYTEK_자료/fr3_sample/FR3_API_Reference.md, 그쪽도 실기 시험 전 판):
  - 교정표·AAKit 파일이 없다. 채널 위상은 배열 기하로만 계산한다 (문서 11절, 피치 18.5 mm)
  - 한 대라도 반드시 그룹에 넣는다 (setGroup). 그룹이 각 채널이 배열 어느 자리에 꽂혔는지(channel_map)를 안다.
    그룹 정보는 TLKCore(호스트) 안에만 있으므로 BBox 를 만지는 프로세스마다 다시 만든다 → setup_bbox
  - 여기서는 BBox 한 대 = 그룹 하나 (group_id 1~8 이 한 TLKCore 안에서 겹치면 안 된다. 우리는 7대).
    위상은 자기 channel_map 만으로 계산되므로 RU 두 대를 한 그룹에 넣는 것과 결과가 같다
  - 주파수는 BBox 가 스스로 맞추지 않고 위상 계산에만 쓴다 → setRFFreq 를 실제 RF(7.4 GHz)로. 틀리면 빔이 엉뚱한 데를 본다
  - 이득 0~31 dB (감쇠기, att = 31 − gain, 0.1 dB 단위). "최대 − backoff" 의 최대는 31
  - TX 설정과 RX 설정이 따로 있다. setBeamAngle 에 rf_mode 를 늘 준다 (안 주면 장치에 TDD? 를 한 번 더 묻는다)
  - setBeamAngle 은 꽂힌 채널을 다시 켠다 → switch_channels(off=True) 뒤에 set_beam 을 부르면 그 BBox 는 다시 켜진다
  - RU 안 B 쪽 위상 보정 dphi: setBeamAngle(명령 1번) 뒤에 setChannelPhase 4번으로 기하 위상 + dphi 를 덮어쓴다.
    setBficConfig 는 채널마다 명령 3개(12번)라 10 Hz 추적에 무거워 안 쓴다
  - y = 1·3 줄 V 소자의 180° 회전은 TLKCore 가 setBeamAngle·setChannelPhase 모두에서 알아서 보정한다 (손으로 더하지 않는다)
UD Box (UDB) 쪽: FR3 문서에 없다. 옛 TLKCore 문서대로 setUDFreq(sn, LO, RF, IF, BW) [kHz] 등. ⚠ 실기 미검증
  기준 단자는 입력이나 출력 중 하나만 (TMYTEK 11-17):
  마스터 = setRefSource(INTERNAL) + setOutputReference(True, 10 MHz) · 나머지 = setRefSource(EXTERNAL, 10 MHz)

config 의 BBox 항목 (demo_config.json tmytek.bbox / rx_bbox):
  sn · row(upper/lower/ue) · dphi_deg · (선택) ip — 주면 스캔 없이 initDev(sn, ip, 33)
  (선택) chmap — [[x, y, "H"|"V"], ...] ch1~ch4 가 꽂힌 자리. 좌표는 배열 **뒤(커넥터 쪽)에서 본** 왼쪽 위 (0,0),
  x 오른쪽(방위) · y 아래(고각). 안 주면 row 로 정한다: upper = y 0 줄 · lower = y 1 줄 · ue = y 0 줄 (모두 H 1×4)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

FR3_DEV_TYPE = 33
FR3_GAIN_MAX = 31.0
FR3_PITCH_MM = 18.5
C_MM_PER_S = 3.0e11                               # TLKCore 가 c = 3e8 m/s 로 계산한다 (11-03 로그 θ45° 7.1 GHz → 111.454°)

DEFAULT_CHMAP = {
    "upper": [[0, 0, "H"], [1, 0, "H"], [2, 0, "H"], [3, 0, "H"]],
    "lower": [[0, 1, "H"], [1, 1, "H"], [2, 1, "H"], [3, 1, "H"]],
    "ue":    [[0, 0, "H"], [1, 0, "H"], [2, 0, "H"], [3, 0, "H"]],
}


def _log(msg):
    print(msg, flush=True)


def chmap_of(box: dict | None) -> list[list]:
    """config 의 BBox 항목 → channel_map [[x, y, pol], ...]."""
    box = box or {}
    if box.get("chmap"):
        return [[int(c[0]), int(c[1]), str(c[2]).upper()] for c in box["chmap"]]
    return DEFAULT_CHMAP.get(box.get("row", "upper"), DEFAULT_CHMAP["upper"])


def steer_phases(chmap, freq_hz: float, theta_deg: float, phi_deg: float, pitch_mm: float = FR3_PITCH_MM) -> list[float]:
    """FR3 문서 11절 식 그대로 (회전 소자 보정 전 = 소자가 실제로 내는 위상) [deg, 0~360)."""
    lam = C_MM_PER_S / freq_hz
    st = math.sin(math.radians(theta_deg))
    dx = 360.0 * pitch_mm / lam * st * math.cos(math.radians(phi_deg))
    dy = -360.0 * pitch_mm / lam * st * math.sin(math.radians(phi_deg))
    return [(x * dx + y * dy) % 360.0 for x, y, _ in chmap]


class TMY:
    def __init__(self, working_root: str | Path = ".", dry_run: bool = False):
        self.dry = dry_run
        self.root = Path(working_root).resolve()
        self.svc = None
        self.types = {}
        self.dr = {}                                      # sn → 이득 범위 (0, 31)
        self.T = None                                     # tlkcore 모듈
        self.groups = {}                                  # sn → nano_id
        self.chmap = {}                                   # sn → channel_map
        self.freq_hz = {}                                 # sn → RF [Hz]
        self.mode = {}                                    # sn → "TX" | "RX"
        if not self.dry:
            try:
                lib = self.root / "lib"
                if lib.exists():
                    sys.path.insert(0, str(lib))
                import tlkcore as T                       # noqa: N812
                self.T = T
            except Exception as e:                        # noqa: BLE001
                _log(f"[TMY] tlkcore 를 못 불렀다 ({e!r}) → dry-run 으로 진행")
                self.dry = True
            else:
                if not hasattr(T.TLKCoreService, "setGroupInit"):
                    _log("[TMY] ⚠ 이 tlkcore 에 setGroupInit 이 없다 — BBox Lite FR3(Device Type 33) 를 지원하는 판이 아니다")

    # ---------------- 연결 ----------------
    def open(self, sns: list[str] | None = None, addrs: dict | None = None) -> dict:
        """sns 를 연결한다. addrs = {sn: ip} 를 주면 그 장치는 스캔 없이 initDev(sn, ip, 33) (FR3 만)."""
        addrs = addrs or {}
        if self.dry:
            _log(f"[TMY dry] scan/init {sns}")
            self.types = {sn: "dry" for sn in (sns or [])}
            return self.types
        T = self.T
        try:
            self.svc = T.TLKCoreService(working_root=str(self.root))
        except TypeError:
            self.svc = T.TLKCoreService()
        if not getattr(self.svc, "running", True):
            raise RuntimeError("TLKCoreService 가 안 뜬다")
        # 스캔은 이미 연결된 장치를 모두 풀므로(문서 4절) 직결(addrs)은 스캔 뒤에 한다
        want_scan = not sns or any(sn not in addrs for sn in sns)
        scan = {}
        if want_scan:
            try:
                self.svc.scanDevices(interface=T.DevInterface.ALL)
            except (TypeError, AttributeError):
                self.svc.scanDevices()
            scan = self.svc.getScanInfo().RetData or {}
        for sn, (addr, devtype, in_dfu) in scan.items():
            if (sns and sn not in sns) or sn in addrs:
                continue
            r = self.svc.initDev(sn)
            self.types[sn] = devtype
            _log(f"[TMY] {sn} {addr} type={devtype} dfu={in_dfu} init={r.RetCode}")
        for sn, ip in addrs.items():
            if sns and sn not in sns:
                continue
            r = self.svc.initDev(sn, ip, FR3_DEV_TYPE)
            self.types[sn] = FR3_DEV_TYPE
            _log(f"[TMY] {sn} {ip} type={FR3_DEV_TYPE} (직결) init={r.RetCode}")
        missing = [s for s in (sns or []) if s not in self.types]
        if missing:
            _log(f"[TMY] ⚠ 못 찾은 SN: {missing}")
        return self.types

    def _ok(self, ret, what):
        rc = getattr(ret, "RetCode", None)
        if self.T is not None and rc is not None and rc is not self.T.RetCode.OK:
            raise RuntimeError(f"{what} 실패: {rc.name if hasattr(rc, 'name') else rc} {getattr(ret, 'RetMsg', '')}")
        return ret

    # ---------------- UD Box ----------------
    def setup_udb(self, sn: str, role: str, lo_ghz: float, rf_ghz: float, if_ghz: float, bw_mhz: float,
                  ref_khz: int = 10000):
        lo, rf, ifq, bw = lo_ghz * 1e6, rf_ghz * 1e6, if_ghz * 1e6, bw_mhz * 1e3      # kHz
        if self.dry:
            _log(f"[TMY dry] UDB {sn} role={role} LO {lo_ghz} RF {rf_ghz} IF {if_ghz} GHz BW {bw_mhz} MHz ref {ref_khz} kHz")
            return
        T, s = self.T, self.svc
        if role == "master":
            self._ok(s.setRefSource(sn, T.UD_REF.INTERNAL), f"{sn} ref internal")
            self._ok(s.setOutputReference(sn, True, ref_khz), f"{sn} ref output")
        else:
            self._ok(s.setRefSource(sn, T.UD_REF.EXTERNAL, ref_khz), f"{sn} ref external")
        self._ok(s.setUDFreq(sn, lo, rf, ifq, bw), f"{sn} setUDFreq")
        _log(f"[TMY] UDB {sn} freq={s.getUDFreq(sn)} ref={s.getRefConfig(sn)}")

    def udb_locked(self, sn: str) -> dict:
        if self.dry:
            return {"REF_LOCK": True, "PLO_LOCK": True, "dry": True}
        T = self.T
        ret = self.svc.getUDState(sn, T.UDMState.REF_LOCK | T.UDMState.PLO_LOCK)
        return dict(ret.RetData) if ret.RetData else {"error": str(ret)}

    # ---------------- BBox Lite FR3 ----------------
    # mode = "TX" (RU 6대) | "RX" (UE). TDD 반이중이라 한 BBox 는 한 모드.
    def _mode(self, mode: str):
        return getattr(self.T.CellRFMode, mode.upper())

    def setup_bbox(self, sn: str, freq_ghz: float, aakit_match: str = "4x4", mode: str = "TX", box: dict | None = None):
        """그룹(한 대) 만들기 → RF 주파수 → RF 모드. aakit_match 는 옛 판 호환용 (FR3 에는 AAKit 파일이 없다).
        돌려주는 값 = 이득 범위 (0, 31)."""
        cm = chmap_of(box)
        self.chmap[sn], self.freq_hz[sn], self.mode[sn] = cm, float(freq_ghz) * 1e9, mode.upper()
        self.dr[sn] = (0.0, FR3_GAIN_MAX)
        if self.dry:
            _log(f"[TMY dry] BBox {sn} {mode} · {freq_ghz} GHz · chmap {cm}")
            return self.dr[sn]
        s = self.svc
        if sn not in self.groups:
            gid = len(self.groups) + 1
            if gid > 8:
                raise RuntimeError("group_id 는 1~8 — 한 프로세스에서 BBox 9대 이상은 그룹을 묶어야 한다")
            nano = f"FR3G{gid:04d}"
            self._ok(s.setGroup(nano, gid, {sn: tuple(tuple(c) for c in cm)}), f"{sn} setGroup {nano}")
            self.groups[sn] = nano
        self._ok(s.setRFFreq(sn, int(round(self.freq_hz[sn]))), f"{sn} setRFFreq {freq_ghz} GHz")
        self._ok(s.setRFMode(sn, self._mode(mode)), f"{sn} setRFMode {mode}")
        _log(f"[TMY] BBox {sn} {mode} · {freq_ghz} GHz · 그룹 {self.groups[sn]} · chmap {cm}")
        return self.dr[sn]

    def max_gain(self, sn: str, mode: str = "TX"):
        """장치 이득 범위. FR3 는 0~31 dB 고정."""
        return self.dr.get(sn, (0.0, FR3_GAIN_MAX))

    def _angle(self, theta_deg: float, phi_deg: float):
        """수평 조향(phi 0/180)이면 AzElAngle(az=±theta, el=0) — 고각 0 이면 어느 규약이든 방위가 같다.
        그 밖이면 ThetaPhiAngle (정수 각)."""
        T = self.T
        sx = math.sin(math.radians(theta_deg)) * math.cos(math.radians(phi_deg))
        sy = math.sin(math.radians(theta_deg)) * math.sin(math.radians(phi_deg))
        if abs(sy) < 1e-9:
            return T.AzElAngle(azimuth=math.degrees(math.asin(max(-1.0, min(1.0, sx)))), elevation=0.0)
        return T.ThetaPhiAngle(theta=int(round(theta_deg)), phi=int(round(phi_deg)) % 360)

    def set_beam(self, sn: str, gain_db: float, theta_deg: float, phi_deg: float, dphi_deg: float = 0.0):
        """theta = 정면에서 벌어진 각, phi = 배열 평면 안 방향 (geometry.to_theta_phi). dphi = RU 안 B 쪽 보정 위상.
        setup_bbox 를 먼저 불러야 한다 (그룹·주파수·모드)."""
        if self.dry:
            return
        if sn not in self.groups:
            raise RuntimeError(f"{sn}: setup_bbox 를 먼저 (FR3 는 그룹이 있어야 빔을 계산한다)")
        s, mode = self.svc, self._mode(self.mode[sn])
        g = max(0.0, min(FR3_GAIN_MAX, float(gain_db)))
        self._ok(s.setBeamAngle(sn, self._angle(theta_deg, phi_deg), g, rf_mode=mode), f"{sn} setBeamAngle")
        if abs(dphi_deg) > 1e-6:
            ph = steer_phases(self.chmap[sn], self.freq_hz[sn], theta_deg, phi_deg)
            for ch, p in enumerate(ph, start=1):
                self._ok(s.setChannelPhase(sn, mode, ch, (p + dphi_deg) % 360.0), f"{sn} ch{ch} setChannelPhase(dphi)")

    def switch_channels(self, sn: str, off: bool):
        """RU 안 교정용: 한 BBox 의 꽂힌 채널을 모두 끄거나 켠다 (현재 모드의 설정)."""
        if self.dry:
            _log(f"[TMY dry] BBox {sn} channels {'OFF' if off else 'ON'}")
            return
        mode = self._mode(self.mode.get(sn, "TX"))
        for ch in range(1, len(self.chmap.get(sn, DEFAULT_CHMAP["upper"])) + 1):
            self._ok(self.svc.setChannelEnable(sn, mode, ch, not off), f"{sn} ch{ch} setChannelEnable")

    def find_me(self, sn: str, on: bool = True):
        """현장에서 SN 과 실물을 맞출 때: LED 깜빡임."""
        if self.dry:
            _log(f"[TMY dry] find me {sn} {on}")
            return
        self._ok(self.svc.setFindMe(sn, on), f"{sn} setFindMe")

    def close(self):
        if self.svc is None:
            return
        for nano in set(self.groups.values()):
            try:
                self.svc.delGroup(nano)
            except Exception:                              # noqa: BLE001
                pass
        for sn in self.types:
            try:
                self.svc.DeInitDev(sn)
            except Exception:                              # noqa: BLE001
                pass


def gain_for(dr, gain_db=None, backoff_db: float = 0.0):
    """gain_db 를 주면 그 값, 아니면 장치 최대 − backoff_db (범위 아래로는 안 내림).
    FR3 최대 = 31 (감쇠 0). ⚠ TX 최대면 수신 BBox 가 포화한다 (IP1dB −32 dBm) → config 의 tx_backoff_db."""
    if gain_db is not None:
        return float(gain_db)
    try:
        lo, hi = float(dr[0]), float(dr[1])
        return max(lo, hi - float(backoff_db))
    except Exception:                                      # noqa: BLE001
        return 0.0


def cfg_gain(tcfg: dict, dr, kind: str = "tx") -> float:
    """config['tmytek'] 에서 kind(tx/rx) 이득: {kind}_gain_db 가 있으면 그 값, 없으면 최대 − {kind}_backoff_db."""
    return gain_for(dr, tcfg.get(f"{kind}_gain_db"), tcfg.get(f"{kind}_backoff_db", 0.0))


def addrs_of(boxes: list[dict]) -> dict:
    """config 의 BBox 항목 중 ip 가 있는 것 → {sn: ip} (스캔 없이 직결)."""
    return {b["sn"]: b["ip"] for b in boxes if b.get("ip")}


def setup_all_bbox(tmy: "TMY", cfg: dict, rx: bool = True, broadside: bool = True) -> dict:
    """TX BBox 6대(+ RX BBox)를 이 프로세스에서 설정하고 {sn: 이득} 을 돌려준다.
    FR3 그룹·주파수는 TLKCore(호스트) 안의 상태라 BBox 를 만지는 프로세스마다 한 번 해야 한다."""
    t, rf = cfg["tmytek"], cfg["freq"]["rf_ghz"]
    gains = {}
    for b in t["bbox"]:
        dr = tmy.setup_bbox(b["sn"], rf, mode="TX", box=b)
        gains[b["sn"]] = cfg_gain(t, dr, "tx")
        if broadside:
            tmy.set_beam(b["sn"], gains[b["sn"]], 0.0, 0.0, b.get("dphi_deg", 0.0))
    for b in (t.get("rx_bbox", []) if rx else []):
        dr = tmy.setup_bbox(b["sn"], rf, mode="RX", box=b)
        gains[b["sn"]] = cfg_gain(t, dr, "rx")
        if broadside:
            tmy.set_beam(b["sn"], gains[b["sn"]], 0.0, 0.0, b.get("dphi_deg", 0.0))
    return gains
