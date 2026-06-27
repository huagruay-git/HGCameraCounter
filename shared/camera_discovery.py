"""
Camera / IP discovery engine — find IP cameras of ALL kinds on the local network.

Dependency-free (stdlib only; psutil used if present for nicer interface listing).
Shared by the GUI LAN scanner (controller/camera_manager.py) and the headless CLI
(scripts/discover_cameras.py).

What it does, layered from broad to specific:
  1. list_local_subnets()  — every local IPv4 /24 the PC is attached to (so a PC with a
     2nd NIC on the camera VLAN is covered, not just the default-route subnet).
  2. onvif_ws_discovery()  — ONVIF WS-Discovery multicast probe: cameras announce
     themselves with NO IP/path/credentials needed. Finds "all kinds" of ONVIF cams.
  3. scan()                — TCP-probe each host on common RTSP/camera ports, then try a
     multi-brand list of RTSP paths and validate with an RTSP OPTIONS handshake
     (200 = open, 401 = needs auth → still a real camera). Brand is guessed from the
     matching path. ONVIF-found hosts are folded in as priority targets.

Everything is best-effort and never raises out of the public helpers.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

# Common camera / NVR ports. 554/8554 carry RTSP; the rest just help flag a device.
DEFAULT_PORTS: List[int] = [554, 8554, 88, 10554]

# Multi-brand RTSP path templates. "{channel}" is expanded over the channel range.
# Order matters: most specific / most common first (we stop at the first 200/401).
BRAND_PATHS: List[Tuple[str, str]] = [
    ("Dahua/Imou", "/cam/realmonitor?channel={channel}&subtype=0"),
    ("Hikvision",  "/Streaming/Channels/{channel}01"),
    ("Hikvision",  "/ISAPI/Streaming/channels/{channel}01"),
    ("Hik/legacy", "/h264/ch{channel}/main/av_stream"),
    ("Uniview",    "/unicast/c{channel}/s0/live"),
    ("Axis",       "/axis-media/media.amp"),
    ("Vivotek",    "/live.sdp"),
    ("Generic",    "/live/ch{channel}_0"),
    ("Generic",    "/live{channel}.sdp"),
    ("Generic",    "/stream{channel}"),
    ("Generic",    "/11"),
    ("Generic",    "/"),
]


# ---------------------------------------------------------------------------
# Local interfaces / subnets
# ---------------------------------------------------------------------------
def list_local_ips() -> List[str]:
    """Best-effort list of this host's IPv4 addresses (excludes loopback/link-local)."""
    ips: set[str] = set()
    # Preferred: psutil enumerates every interface.
    try:
        import psutil  # type: ignore
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if getattr(a, "family", None) == socket.AF_INET and a.address:
                    ips.add(a.address)
    except Exception:
        pass
    # Fallbacks (always run; cheap and fills gaps in headless/frozen envs).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    return sorted(
        ip for ip in ips
        if ip and not ip.startswith("127.") and not ip.startswith("169.254.")
    )


def list_local_subnets() -> List[str]:
    """Every local /24 (CIDR) derived from this host's IPv4 addresses."""
    nets: List[str] = []
    seen: set[str] = set()
    for ip in list_local_ips():
        octets = ip.split(".")
        if len(octets) != 4:
            continue
        cidr = f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
        if cidr not in seen:
            seen.add(cidr)
            nets.append(cidr)
    return nets or ["192.168.1.0/24"]


# ---------------------------------------------------------------------------
# ONVIF WS-Discovery (multicast) — finds cameras with no IP/creds needed
# ---------------------------------------------------------------------------
_WSD_ADDR = "239.255.255.250"
_WSD_PORT = 3702
_WSD_PROBE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
    'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
    'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
    'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    "<e:Header>"
    "<w:MessageID>uuid:hgcc-probe-0001-0001-0001-000000000001</w:MessageID>"
    "<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
    "<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>"
    "</e:Header>"
    "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>"
    "</e:Envelope>"
)


def _is_ipv4(host: str) -> bool:
    try:
        ipaddress.IPv4Address(host)
        return True
    except Exception:
        return False


def onvif_ws_discovery(timeout: float = 2.5) -> List[Dict[str, str]]:
    """Send a WS-Discovery Probe out every local interface; collect ONVIF responders.

    Returns a list of {"ip", "xaddr"} for each discovered ONVIF camera. Only real
    IPv4 hosts are kept (XML namespace URLs and non-ONVIF WSD responders such as
    Windows PCs are filtered out, as are this host's own addresses).
    """
    found: Dict[str, str] = {}  # ip -> first xaddr
    local_set = set(list_local_ips())
    local_ips = sorted(local_set) or [""]
    for local_ip in local_ips:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if local_ip:
                try:
                    sock.bind((local_ip, 0))
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                    socket.inet_aton(local_ip))
                except Exception:
                    pass
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.settimeout(timeout)
            sock.sendto(_WSD_PROBE.encode("utf-8"), (_WSD_ADDR, _WSD_PORT))

            import time as _t
            end = _t.time() + timeout
            while _t.time() < end:
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    break
                except Exception:
                    break
                text = data.decode("utf-8", errors="ignore")
                if "onvif" not in text.lower():
                    continue  # a non-ONVIF WSD responder (e.g. a Windows PC) — skip
                # Only trust URLs inside <...XAddrs>...</...XAddrs> (the device service),
                # not the XML namespace URIs scattered through the envelope.
                for block in re.findall(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>",
                                        text, re.IGNORECASE | re.DOTALL):
                    for xaddr in re.findall(r"https?://[^\s<>\"]+", block):
                        host = urlparse(xaddr).hostname
                        if host and _is_ipv4(host) and host not in local_set \
                                and host not in found:
                            found[host] = xaddr
                # Devices replying without XAddrs are still a hit via their source IP.
                src = addr[0] if addr else None
                if src and _is_ipv4(src) and src not in local_set and src not in found:
                    found.setdefault(src, "")
        except Exception:
            pass
        finally:
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
    return [{"ip": ip, "xaddr": xa} for ip, xa in found.items()]


# ---------------------------------------------------------------------------
# Low-level probes
# ---------------------------------------------------------------------------
def tcp_open(ip: str, port: int, timeout_ms: int) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout_ms / 1000.0):
            return True
    except Exception:
        return False


def rtsp_options(rtsp_url: str, timeout_ms: int) -> Tuple[Optional[int], str]:
    """RTSP OPTIONS handshake. Returns (status_code, first_line). 200=open, 401=auth."""
    timeout_sec = timeout_ms / 1000.0
    try:
        parsed = urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return None, "invalid host"
        request = (
            f"OPTIONS {rtsp_url} RTSP/1.0\r\n"
            "CSeq: 1\r\n"
            "User-Agent: HGCameraCounter/1.0\r\n"
            "\r\n"
        ).encode("ascii", errors="ignore")
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.settimeout(timeout_sec)
            sock.sendall(request)
            data = sock.recv(4096)
        if not data:
            return None, "no response"
        first = data.decode("utf-8", errors="ignore").splitlines()[0].strip()
        m = re.search(r"RTSP/\d\.\d\s+(\d{3})", first)
        return (int(m.group(1)), first) if m else (None, first or "invalid header")
    except Exception as e:
        return None, str(e)


def build_rtsp_url(ip: str, port: int, path: str, username: str, password: str) -> str:
    auth = ""
    if username or password:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    p = path if path.startswith("/") else f"/{path}"
    return f"rtsp://{auth}{ip}:{port}{p}"


def candidate_paths(channel_start: int, channel_end: int,
                    extra_template: str = "", all_brands: bool = True) -> List[Tuple[str, str]]:
    """Return [(brand, path)] to try, across the channel range.

    extra_template (if any) is tried first; all_brands appends the multi-brand list.
    If both would yield nothing, brand paths are used anyway so we never probe empty.
    """
    lo, hi = min(channel_start, channel_end), max(channel_start, channel_end)
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add(brand: str, tmpl: str):
        if "{channel}" in tmpl:
            for ch in range(lo, hi + 1):
                p = tmpl.replace("{channel}", str(ch))
                if p not in seen:
                    seen.add(p); out.append((brand, p))
        else:
            if tmpl not in seen:
                seen.add(tmpl); out.append((brand, tmpl))

    if extra_template.strip():
        add("Custom", extra_template.strip())
    if all_brands:
        for brand, tmpl in BRAND_PATHS:
            add(brand, tmpl)
    if not out:  # neither a custom template nor brands -> fall back to brands
        for brand, tmpl in BRAND_PATHS:
            add(brand, tmpl)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _expand_hosts(subnets: List[str], cap: int = 8192) -> List[str]:
    hosts: List[str] = []
    seen: set[str] = set()
    for sn in subnets:
        sn = sn.strip()
        if not sn:
            continue
        try:
            if "/" in sn:
                net = ipaddress.ip_network(sn, strict=False)
                ips = [str(h) for h in net.hosts()] if net.num_addresses > 1 else [str(net.network_address)]
            else:
                ips = [str(ipaddress.ip_address(sn))]  # bare IP
        except ValueError:
            continue
        for ip in ips:
            if ip not in seen:
                seen.add(ip); hosts.append(ip)
                if len(hosts) >= cap:
                    return hosts
    return hosts


def probe_host(ip: str, ports: List[int], paths: List[Tuple[str, str]],
               username: str, password: str, timeout_ms: int,
               known_camera: bool = False) -> Optional[Dict]:
    """Probe one host. Returns a match dict or None.

    known_camera=True (e.g. ONVIF-discovered) still reports the host even if no RTSP
    path matched, so the user can fix the path/creds manually.
    """
    for port in ports:
        if not tcp_open(ip, port, timeout_ms):
            continue
        first_resp: Optional[Tuple[int, str]] = None
        for brand, path in paths:
            url = build_rtsp_url(ip, port, path, username, password)
            code, line = rtsp_options(url, timeout_ms)
            if code is None:
                continue
            if first_resp is None:
                first_resp = (code, line)
            if code in (200, 401):
                return {
                    "ip": ip, "port": port, "rtsp_url": url, "brand": brand,
                    "status_code": code, "status_text": line,
                    "auth_required": code == 401,
                    "note": "RTSP open" if code == 200 else "auth required",
                }
        if first_resp is not None:
            code, line = first_resp
            return {
                "ip": ip, "port": port, "rtsp_url": "", "brand": "Unknown",
                "status_code": code, "status_text": line, "auth_required": False,
                "note": "RTSP service detected but no known path matched",
            }
    if known_camera:
        return {
            "ip": ip, "port": ports[0] if ports else 554, "rtsp_url": "",
            "brand": "ONVIF", "status_code": None, "status_text": "",
            "auth_required": False,
            "note": "ONVIF device found (no RTSP path confirmed — set path/creds manually)",
        }
    return None


def scan(
    subnets: List[str],
    username: str = "",
    password: str = "",
    ports: Optional[List[int]] = None,
    channel_start: int = 1,
    channel_end: int = 1,
    extra_template: str = "",
    all_brands: bool = True,
    timeout_ms: int = 1200,
    max_workers: int = 64,
    use_onvif: bool = True,
    on_found: Optional[Callable[[Dict], None]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """Discover cameras across subnets + ONVIF. Returns a deduped list of match dicts."""
    ports = ports or list(DEFAULT_PORTS)
    paths = candidate_paths(channel_start, channel_end, extra_template, all_brands)

    onvif_ips: set[str] = set()
    if use_onvif:
        for dev in onvif_ws_discovery():
            onvif_ips.add(dev["ip"])
        if on_progress:
            on_progress(0, 1, f"ONVIF found {len(onvif_ips)} device(s)")

    hosts = _expand_hosts(list(subnets))
    # ONVIF hits first (priority), then the rest of the subnet hosts.
    ordered = [ip for ip in onvif_ips] + [ip for ip in hosts if ip not in onvif_ips]
    total = len(ordered) or 1

    results: List[Dict] = []
    seen_keys: set[str] = set()
    done = 0

    def _job(ip: str) -> Optional[Dict]:
        if should_stop and should_stop():
            return None
        return probe_host(ip, ports, paths, username, password, timeout_ms,
                          known_camera=(ip in onvif_ips))

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 256))) as pool:
        futures = {pool.submit(_job, ip): ip for ip in ordered}
        for fut in as_completed(futures):
            if should_stop and should_stop():
                for f in futures:
                    f.cancel()
                break
            ip = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                key = f"{res['ip']}:{res['port']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(res)
                    if on_found:
                        on_found(res)
                if on_progress:
                    on_progress(done, total, f"{ip} found")
            elif on_progress and (done == total or done % 16 == 0):
                on_progress(done, total, f"{ip} scanned")

    def _ip_key(r: Dict) -> Tuple[int, int, int, int]:
        try:
            return tuple(int(x) for x in str(r["ip"]).split("."))  # type: ignore[return-value]
        except Exception:
            return (999, 999, 999, 999)

    results.sort(key=_ip_key)
    return results
