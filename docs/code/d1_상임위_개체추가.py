import re
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, List, DefaultDict
from collections import defaultdict

from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS, XSD, FOAF, OWL

# ====== 상임위 값(자동 감지로 지연 초기화) ======
COMMITTEE_CODE: Optional[str]     = None
COMMITTEE_LABEL_KO: Optional[str] = None
BRANCH: Optional[str]             = None
COMMITTEE_IRI: Optional[str]      = None

# ====== 경로/입력 ======
CSV_PATH       = r"D:/연구/국회회의록/병합/251025_국운위_병합.csv"         # 상임위 CSV
OUT_TTL        = r"D:/연구/국회회의록/251105_SC/d1-record_SC-{}.ttl"      # 출력 TTL(d1) (자동 치환됨)
PEOPLE_TTL     = r"https://lseun135.github.io/NAontology_study/resource/d2-person.ttl"
OUT_TMP_PEOPLE = r"D:/연구/국회회의록/251105_SC/tmp_person_SC_{}.ttl"     # 임시 Person TTL (자동 치환됨)
EXISTING_D1_TTL= r"https://lseun135.github.io/NAontology_study/resource/d1-record.ttl"  # 기존 d1 TTL(중복 방지)
CSV_ENCODING   = "utf-8-sig"

# ====== Namespaces ======
NAM1   = Namespace("https://lseun135.github.io/NAontology_study/m1-record#")
NAM2   = Namespace("https://lseun135.github.io/NAontology_study/m2-person#")
NAM3   = Namespace("https://lseun135.github.io/NAontology_study/m3-content#")

NADAT1 = Namespace("https://lseun135.github.io/NAontology_study/resource/d1-record#")
NADAT2 = Namespace("https://lseun135.github.io/NAontology_study/resource/d2-person#")
NADAT3 = Namespace("https://lseun135.github.io/NAontology_study/resource/d3-content#")

RICO   = Namespace("https://www.ica.org/standards/RiC/ontology#")
ORG    = Namespace("http://www.w3.org/ns/org#")
SCHEMA = Namespace("https://schema.org/")
SKOS_HTTPS = Namespace("https://www.w3.org/2004/02/skos/core#")  # 방어적 정의(일부 d2 파일 호환)

# ====== 숫자/날짜 유틸 ======
def digits_only(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    m = re.findall(r"\d+", str(s))
    return int("".join(m)) if m else None

def pad(n, width):
    return f"{n:0{width}d}" if n is not None else None

def parse_date_simple(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    nums = re.findall(r"\d+", str(s))
    if len(nums) >= 3:
        y, m, d = map(int, nums[:3])
        if 1 <= m <= 12 and 1 <= d <= 31:
            return y, m, d
    return None

def ymd_to_short_id(y:int, m:int, d:int) -> str:
    return f"{y%100:02d}{m:02d}{d:02d}"

def ymd_to_iso(y:int, m:int, d:int) -> str:
    return date(y, m, d).isoformat()

# >>> 날짜 표현(ko) & 단일 Date 노드 보장
KOR_WEEKDAY = ["월","화","수","목","금","토","일"]

def expressed_date_ko(y: int, m: int, d: int) -> str:
    wd = date(y, m, d).weekday()  # 0=월 ~ 6=일
    return f"{y}년{m}월{d}일({KOR_WEEKDAY[wd]})"

def ensure_single_date_node(g: Graph, seen: dict, y: int, m: int, d: int) -> URIRef:
    short_id = ymd_to_short_id(y, m, d)
    date_id  = f"date-{short_id}"
    node     = URIRef(NADAT1[date_id])

    if date_id not in seen["date"]:
        seen["date"].add(date_id)
        g.add((node, RDF.type, RICO.Date))
        g.add((node, RICO.identifier, Literal(date_id)))
        g.add((node, RICO.name,       Literal(date_id)))
        g.add((node, RICO.normalizedDateValue, Literal(ymd_to_iso(y, m, d), datatype=XSD.date)))
        g.add((node, RICO.expressedDate, Literal(expressed_date_ko(y, m, d), lang="ko")))
        g.add((node, RICO.hasDateType, URIRef(str(NAM1["singleDate"]))))

    return node
# <<<

# ====== d1 마스터(+imports) 파서 ======
def parse_with_imports(main_ttl_url_or_path: str) -> Graph:
    g = Graph()
    g.parse(main_ttl_url_or_path, format="turtle")
    imports = set()
    for _s, _p, o in g.triples((None, OWL.imports, None)):
        if isinstance(o, URIRef):
            imports.add(str(o))
    for imp in sorted(imports):
        try:
            imp_fixed = re.sub(r"/resource/d1-record_SC-([A-Z]{2})\.ttl$",
                               r"/resource/d1-record/d1_SC-\1.ttl", imp)
            g.parse(imp_fixed, format="turtle")
        except Exception as e:
            print(f"[warn] parse_with_imports: import 파싱 실패: {imp} ({e})")
    return g

# ====== 기존 d1 TTL에서 identifier 선점(중복 방지) ======
def load_existing_identifiers(d1_ttl_path: str) -> set:
    ids = set()
    if not d1_ttl_path:
        return ids
    try:
        g = parse_with_imports(d1_ttl_path)  # imports까지 포함
        for _s, _p, o in g.triples((None, RICO.identifier, None)):
            if isinstance(o, Literal):
                ids.add(str(o))
    except Exception as e:
        print(f"[warn] load_existing_identifiers: {e}")
    return ids

# ====== d1에서 rico:Date만 별도 수집(중복 진단) ======
def load_existing_date_ids(d1_ttl_path: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not d1_ttl_path:
        return result
    try:
        g = parse_with_imports(d1_ttl_path)
        for s in g.subjects(RDF.type, RICO.Date):
            ids = [str(o) for o in g.objects(s, RICO.identifier) if isinstance(o, Literal)]
            if not ids:
                continue
            for idv in ids:
                result.setdefault(idv, []).append(str(s))
    except Exception as e:
        print(f"[warn] load_existing_date_ids: {e}")
    return result

def map_date_ids_per_import(main_ttl_url_or_path: str) -> Dict[str, Dict[str, List[str]]]:
    per_file: Dict[str, Dict[str, List[str]]] = {}
    try:
        master = Graph()
        master.parse(main_ttl_url_or_path, format="turtle")
        imports = [str(o) for _s, _p, o in master.triples((None, OWL.imports, None)) if isinstance(o, URIRef)]
        files = [main_ttl_url_or_path] + imports
        for f in files:
            try:
                f_fixed = re.sub(r"/resource/d1-record_SC-([A-Z]{2})\.ttl$",
                                 r"/resource/d1-record/d1_SC-\1.ttl", f)
                g = Graph()
                g.parse(f_fixed, format="turtle")
                imap: Dict[str, List[str]] = {}
                for s in g.subjects(RDF.type, RICO.Date):
                    ids = [str(o) for o in g.objects(s, RICO.identifier) if isinstance(o, Literal)]
                    for idv in ids:
                        imap.setdefault(idv, []).append(str(s))
                per_file[f_fixed] = imap
            except Exception as ie:
                print(f"[warn] map_date_ids_per_import: 파일 파싱 실패: {f} ({ie})")
    except Exception as e:  
        print(f"[warn] map_date_ids_per_import: {e}")
    return per_file

# ====== (보조) 회기 시작연도 파싱 ======
def parse_session_start_year_from_normalized(g_master: Graph, session_num: int) -> Optional[int]:
    """
    existing_d1_ttl(및 imports)에서 date-session-<회수>의 normalizedDateValue를 찾아
    'YYYY-MM-DD/YYYY-MM-DD' 중 앞쪽 YYYY를 반환. 없으면 None.
    """
    subj = URIRef(NADAT1[f"date-session-{session_num}"])
    for lit in g_master.objects(subj, RICO.normalizedDateValue):
        if isinstance(lit, Literal):
            s = str(lit)
            m = re.match(r"(\d{4})-\d{2}-\d{2}\s*/", s)
            if m:
                return int(m.group(1))
    return None

# ====== RDF 헬퍼 ======
def add_lit(g, s, p, value, lang=None, dt=None):
    if value is None or (isinstance(value, float) and pd.isna(value)): return
    g.add((s, p, Literal(value, lang=lang, datatype=dt)))

def add_res(g, s, p, iri_str):
    if not iri_str: return
    g.add((s, p, URIRef(iri_str)))

# ====== d2-person.ttl → 이름 인덱스 ======
def build_person_index_from_ttl(people_ttl_url):
    idx_exact, idx_nospace = {}, {}
    if not people_ttl_url: return idx_exact, idx_nospace
    gp = Graph()
    # 일부 d2에 time: prefix 누락 가능 → 방어적 보정
    try:
        gp.parse(people_ttl_url, format="turtle")
    except Exception:
        with open("tmp_d2.ttl", "w", encoding="utf-8") as _f:
            _f.write('@prefix time: <http://www.w3.org/2006/time#> .\n')
        gp.parse("tmp_d2.ttl", format="turtle")
        gp.parse(people_ttl_url, format="turtle")

    tmp = {}
    def collect(o_lit, s):
        if isinstance(o_lit, Literal) and (o_lit.language in ("ko", None)):
            name = str(o_lit).strip()
            if name:
                tmp.setdefault(name, set()).add(str(s))
    for s, o in gp.subject_objects(FOAF.name):  collect(o, s)
    for s, o in gp.subject_objects(RDFS.label): collect(o, s)

    for name, iris in tmp.items():
        if len(iris) == 1:
            iri = next(iter(iris))
            idx_exact[name] = iri
            idx_nospace[name.replace(" ", "")] = iri
    return idx_exact, idx_nospace

# ====== d2에서 기존 임시개체(PER-TMP-#####)의 마지막 번호 탐색 ======
def get_last_provisional_number(people_ttl_url: Optional[str]) -> int:
    if not people_ttl_url:
        return 0
    try:
        g = Graph()
        g.parse(people_ttl_url, format="turtle")
        max_n = 0
        pat = re.compile(r"PER-TMP-(\d+)$")
        for s in g.subjects(RDF.type, FOAF.Person):
            m = pat.search(str(s))
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        return max_n
    except Exception as e:
        print(f"[warn] get_last_provisional_number: {e}")
        return 0

# ====== 상임위 자동 인식(d2 + CSV) ======
def build_committee_index_from_ttl(people_ttl_url: str):
    """
    d2(person).ttl에서 상임위 개체(nam2:StandingCommittee)를 수집:
    - label(ko) -> code(CMIT-XX)
    - code -> (label, iri)
    (일부 파일은 SKOS가 https로 정의되어 있으므로 http/https 모두 조회)
    """
    if not people_ttl_url:
        raise ValueError("people_ttl_url이 비어 있습니다.")
    g = Graph()
    g.parse(people_ttl_url, format="turtle")

    idx_label_to_code = {}
    idx_code_to_meta  = {}

    label_preds = [SKOS.prefLabel, SKOS_HTTPS.prefLabel, RDFS.label]

    for s in g.subjects(RDF.type, NAM2["StandingCommittee"]):
        label_ko = None
        code     = None

        for lp in label_preds:
            for o in g.objects(s, lp):
                if isinstance(o, Literal):
                    if (o.language in (None, "ko")):
                        label_ko = str(o).strip()
                        break
            if label_ko:
                break

        for o in g.objects(s, ORG.identifier):
            if isinstance(o, Literal):
                code = str(o).strip()
                break

        if label_ko and code:
            idx_label_to_code[label_ko] = code
            idx_code_to_meta[code] = (label_ko, str(s))

    if not idx_label_to_code:
        raise ValueError("d2에서 상임위 정보를 찾을 수 없습니다. (타입 또는 라벨/식별자 탐색 실패)")
    return idx_label_to_code, idx_code_to_meta

def guess_committee_label_from_df(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cand_cols = [c for c in df.columns if any(k in str(c) for k in ["위원회", "상임위", "위원회명", "소관"])]
    values = []
    for col in cand_cols:
        ser = df[col].dropna().astype(str).str.strip()
        ser = ser[ser.str.contains(r"[가-힣]", regex=True)]
        ser = ser[ser.str.len() <= 30]
        values.extend(ser.head(500).tolist())
    if not values:
        return None
    from collections import Counter
    [(label, _count)] = Counter(values).most_common(1)
    return label

def init_committee_from_sources(df: pd.DataFrame, people_ttl_url: str):
    global COMMITTEE_CODE, COMMITTEE_LABEL_KO, BRANCH, COMMITTEE_IRI

    idx_label_to_code, idx_code_to_meta = build_committee_index_from_ttl(people_ttl_url)
    label_guess = guess_committee_label_from_df(df)
    if not label_guess:
        raise ValueError("CSV에서 상임위 명칭을 찾지 못했습니다. 컬럼(위원회/상임위/위원회명/소관)을 확인하세요.")
    code = idx_label_to_code.get(label_guess)
    if not code:
        for k, v in idx_label_to_code.items():
            if (label_guess in k) or (k in label_guess):
                code = v
                label_guess = k
                break
    if not code:
        raise ValueError(f"d2에서 상임위 코드 매핑을 찾지 못했습니다: '{label_guess}'")
    label, iri = idx_code_to_meta[code]

    COMMITTEE_CODE     = code
    COMMITTEE_LABEL_KO = label
    BRANCH             = code.split("-", 1)[1]
    COMMITTEE_IRI      = iri if iri else str(NADAT2[code])

def rewrite_out_path_with_branch(path_str: str, branch: str) -> str:
    """
    1) 플레이스홀더 우선 치환: "{}", "{BRANCH}", "{branch}"
    2) 없으면 기존 'SC-XX' 패턴을 'SC-{branch}'로 교체
    3) 그래도 없으면 확장자 앞에 '_SC-{branch}' 주입
    """
    if not path_str:
        return path_str
    p = Path(path_str)
    s = str(p)

    if "{" in s and "}" in s:
        new = s.replace("{BRANCH}", branch).replace("{branch}", branch).replace("{}", branch)
        if new != s:
            return new

    new = re.sub(r"SC-[A-Z]{2}", f"SC-{branch}", s)
    if new != s:
        return new

    stem, suf = p.stem, p.suffix
    new_name = f"{stem}_SC-{branch}{suf}" if suf else f"{stem}_SC-{branch}"
    return str(p.with_name(new_name))

# ====== 발언자 파서(이름·역할·식별자 추출) ======
ROLE_KWS = {
    "대통령","국무총리","총리","부총리","부총리겸","의장","부의장",
    "의원","위원","간사","위원장","위원장대리","위원장직무대행","위원장직무대리","후보자",
    "대변인","차관","장관","청장","총장","국장","과장","실장","소장","원장",
    "처장","본부장","센터장","팀장","단장","사무총장","의사국장","직무대리","직무대행","차장","실장","본부장"
}
ROLE_ONLY_TOKENS = {
    "진술인","참고인","증인","발언인","관계자",
    "행정관","담당관","서기관","사무관","연구관","조사관","주무관",
    "자문위원","보좌관","비서관","수석비서관","전문위원","전문관","입법조사관"
}

LOCAL_ROLE_SUFFIXES = (
    "시장","군수","구청장","도지사","교육감",
    "총장","원장","이사장","관장",
    "사장","부사장","회장","협회장","부문장","TF장",
    "차관보","정책관","기획관","조정관","기획조정관",
    "담당관","지원관","교육장","비서관","수석비서관", 
    "관리관","예산관","보건복지관","대변인",
    "사령관","부사령관","참모장","부장","의전장",
    "법률분석관","부대표","정부부대표","사업이사","기획이사","대표이사","상임이사","전무이사",
    "총재","부총재","부총재보","은행장","심의관",
    "감독","조사관","의원비서","비서","변호사","부원장보"
)

ROLE_SUFFIX_RE = re.compile(
    r"(?:[가-힣A-Za-z0-9·\-\(\)㈜]+)?("
    r"대통령|국무총리|총리|부총리(?:겸)?|"
    r"장관|차관|차관보|청장|총장|국장|과장|실장|소장|원장|처장|본부장|센터장|팀장|단장|차장|"
    r"의장|부의장|의원|위원장(?:대리|직무대행|직무대리)?|위원|"
    r"사장|부사장|회장|협회장|이사장|부문장|TF장|"
    r"(?:[가-힣A-Za-z0-9·\-]*정책관)|기획관|조정관|기획조정관|"
    r"(?:[가-힣A-Za-z0-9·\-]*비서관)|수석비서관|비서관|"
    r"관장|"
    r"담당관|지원관|교육장|후보자|임명예정자|"
    r"관리관|예산관|보건복지관|대변인|"
    r"사령관|부사령관|참모장|부장|"
    r"대표이사|"
    r"총재|부총재|부총재보|은행장|심의관|"
    r"감독|조사관|의원비서|비서|"
    r"(?:[가-힣A-Za-z0-9·\-]*상임이사)|상임이사|"
    r"(?:[가-힣A-Za-z0-9·\-]*부대표)|부대표|정부부대표|"
    r"법률분석관|사업이사|기획이사|의전장|"
    r"부원장보|"
    r"직무대리|직무대행"
    r")$"
)

KOREAN_NAME_RE = re.compile(r"^[가-힣]{2,5}$")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\(.*?\)|\[.*?\]|【.*?】|〈.*?〉|《.*?》|「.*?」|『.*?』", "", str(text))
    return re.sub(r"\s+", " ", t).strip()
_strip_brackets_spaces = clean_text

def extract_name_only_from_speaker(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    tokens = clean_text(raw_text).split()
    if not tokens:
        return None

    keep = []
    for t in tokens:
        if (
            (t in ROLE_KWS)
            or ROLE_SUFFIX_RE.search(t)
            or (t in ROLE_ONLY_TOKENS)
            or any(t.endswith(suf) for suf in LOCAL_ROLE_SUFFIXES)
        ):
            continue
        keep.append(t)

    h = [t for t in keep if re.fullmatch(r"[가-힣]{1,5}", t)]
    if not h:
        txt_all = clean_text(raw_text)
        m = re.search(r"([가-힣]{2,5})$", txt_all)
        return m.group(1) if m else None

    cands = []
    for i in range(len(h) - 2):
        s = "".join(h[i:i+3])
        if 2 <= len(s) <= 5 and re.fullmatch(r"[가-힣]{2,5}", s):
            cands.append(s)
    for i in range(len(h) - 1):
        s = h[i] + h[i+1]
        if 2 <= len(s) <= 5 and re.fullmatch(r"[가-힣]{2,5}", s):
            cands.append(s)
    for t in h:
        if 2 <= len(t) <= 5 and re.fullmatch(r"[가-힣]{2,5}", t):
            cands.append(t)

    if not cands:
        txt_all = clean_text(raw_text)
        m = re.search(r"([가-힣]{2,5})$", txt_all)
        return m.group(1) if m else None
    return max(cands, key=len)

def parse_speaker_name(raw, idx_exact, idx_nospace):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None, None
    original = str(raw)
    txt = _strip_brackets_spaces(original)
    if not txt:
        return None, None, None, original
    tokens = txt.split()
    if not tokens:
        return None, None, None, original

    cands = []
    for i in range(len(tokens)):
        cands.append(([i], tokens[i]))
    for i in range(len(tokens) - 1):
        merged = tokens[i] + tokens[i+1]
        if re.fullmatch(r"[가-힣]{2,5}", merged):
            cands.append(([i, i+1], merged))
    for i in range(len(tokens) - 2):
        merged = tokens[i] + tokens[i+1] + tokens[i+2]
        if re.fullmatch(r"[가-힣]{2,5}", merged):
            cands.append(([i, i+1, i+2], merged))

    cands.sort(key=lambda x: len(x[1]), reverse=True)

    for used_idx, cand in cands:
        iri = idx_exact.get(cand) or idx_nospace.get(cand.replace(" ", ""))
        if iri:
            role_tokens = [t for i, t in enumerate(tokens) if i not in used_idx]
            role_text = " ".join(role_tokens) if role_tokens else None
            return cand, iri, role_text, original

    return None, None, None, original

def extract_role_from_speaker(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    txt = clean_text(raw_text)
    if not txt:
        return None
    roles = []
    for t in txt.split():
        if (
            (t in ROLE_KWS)
            or (t in ROLE_ONLY_TOKENS)
            or ROLE_SUFFIX_RE.search(t)
            or any(t.endswith(suf) for suf in LOCAL_ROLE_SUFFIXES)
        ):
            roles.append(t)
    return " ".join(roles) if roles else None

# 원문 순서를 보존해 역할 문자열 뽑기 (이름 앞/뒤 그대로).
def extract_role_phrase_order_sensitive(raw_text: str, name_only: Optional[str]) -> Optional[str]:
    if not raw_text or not name_only:
        return None
    txt = clean_text(str(raw_text))
    name = str(name_only).strip()
    if not txt or not name:
        return None

    idx = txt.find(name)
    if idx == -1:
        return extract_role_from_speaker(raw_text)

    before = txt[:idx].strip()
    after  = txt[idx + len(name):].strip()

    if before and not after:
        role_phrase = before
    elif after and not before:
        role_phrase = after
    elif before and after:
        role_phrase = f"{before} {after}"
    else:
        role_phrase = ""

    role_phrase = re.sub(r"\s+", " ", role_phrase).strip()
    return role_phrase or None

# 괄호표식 추출(예: "(비)" -> "비")
def extract_parenthesis_marker(original_text: str) -> Optional[str]:
    if not original_text:
        return None
    m = re.search(r"\(([^)]+)\)", str(original_text))
    return m.group(1).strip() if m else None

# 발언자 셀의 ' / PER-XX-...' 식별자 추출 (대소문자 허용, 말미 구두점 허용)
def extract_person_code_from_speaker(original_text: str) -> Optional[str]:
    if not original_text:
        return None
    m = re.search(r"/\s*(PER-[A-Za-z]{2}-[A-Za-z0-9]+)\s*[.,;]?\s*$", str(original_text), flags=re.IGNORECASE)
    return m.group(1).upper().strip() if m else None

# ====== 임시 Person 관리 ======
class ProvisionalPersonManager:
    def __init__(self, existing_max_num: int = 0):
        self.counter = existing_max_num  # d2 마지막 번호 이어서 시작
        self.name_to_iri: Dict[str, str] = {}
        # (iri_local, name_only, role_text, marker)
        self.entries: List[Tuple[str, str, Optional[str], Optional[str]]] = []
        # 같은 '이름|LEG'에서 관측된 의원ID들의 집합(동명이인 감지용, 최소 사용)
        self.leg_name_ids: Dict[str, set] = {}

    @staticmethod
    def _norm_key(name_text: str) -> str:
        return (name_text or "").replace(" ", "")

    def get_or_create(self, speaker_cell_text: str, member_id: Optional[str] = None, role_text: Optional[str] = None) -> Optional[str]:
        if not speaker_cell_text:
            return None

        # 1) 이름
        name_only = extract_name_only_from_speaker(speaker_cell_text) or _strip_brackets_spaces(speaker_cell_text)
        base_key = self._norm_key(name_only)

        # 2) 의원ID 정규화(존재 여부만 신뢰)
        mid = ""
        if member_id is not None:
            mid_raw = str(member_id).strip()
            if mid_raw and mid_raw.lower() != "nan":
                mid = mid_raw

        # 3) 키 생성: 의원 여부 우선(LEG), 아니면 '원문 순서의 역할문구'로 분기
        if mid:
            leg_key = f"{base_key}|LEG"
            ids = self.leg_name_ids.setdefault(leg_key, set())
            if not ids or (mid in ids):
                key = leg_key
            else:
                key = f"{leg_key}|ID:{mid}"
            ids.add(mid)
        else:
            role_phrase = extract_role_phrase_order_sensitive(speaker_cell_text, name_only)
            if role_phrase:
                key = f"{base_key}|ROLE:{role_phrase}"
            else:
                key = base_key

        if key in self.name_to_iri:
            return self.name_to_iri[key]

        self.counter += 1
        local_id = f"PER-TMP-{self.counter:05d}"
        iri_str = str(NADAT2[local_id])
        self.name_to_iri[key] = iri_str

        if role_text is None:
            role_text = extract_role_from_speaker(speaker_cell_text)

        marker = extract_parenthesis_marker(speaker_cell_text)
        self.entries.append((local_id, name_only, role_text, marker))
        return iri_str

    def write_ttl_txt(self, out_path: str):
        out_file = Path(out_path).expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append('')
        lines.append('## 임시개체')
        lines.append('')
        for local_id, name_only, role_text, marker in self.entries:
            lines.append(f"nadat2:{local_id} a foaf:Person ;")
            lines.append(f'\trdfs:label "{name_only}"@ko ;')
            lines.append(f'\tfoaf:name "{name_only}"@ko ;')
            if role_text or marker:
                if role_text and marker:
                    comment_txt = f"{role_text} ({marker})"
                elif role_text:
                    comment_txt = role_text
                else:
                    comment_txt = f"({marker})"
                lines.append(f'\trdfs:comment "{comment_txt}"@ko ;')
            lines.append(f"\tnam2:isInAStateOf nam2:Provisional .")
            lines.append('')
        with out_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[write] provisional persons txt: {out_file} (count={len(self.entries)})")

# ====== 메인 변환 ======
def convert(csv_path, out_path, tmp_people_out_path,
            encoding="utf-8-sig", people_ttl_url=None, existing_d1_ttl=""):
    df = pd.read_csv(csv_path, encoding=encoding)

    # --- 상임위 자동 감지 ---
    if people_ttl_url is None:
        raise ValueError("people_ttl_url이 필요합니다(d2 파일 경로/URL).")
    init_committee_from_sources(df, people_ttl_url)
    print(f"[info] committee resolved: code={COMMITTEE_CODE}, label={COMMITTEE_LABEL_KO}, branch={BRANCH}")

    # --- 출력 경로: SC-{BRANCH} 적용 ---
    resolved_out_ttl        = rewrite_out_path_with_branch(out_path, BRANCH)
    resolved_out_tmp_people = rewrite_out_path_with_branch(tmp_people_out_path, BRANCH)

    # --- 그래프 초기화 ---
    g = Graph()
    g.bind("rdf", RDF); g.bind("rdfs", RDFS); g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS); g.bind("xsd", XSD); g.bind("foaf", FOAF); g.bind("owl", OWL)
    g.bind("rico", RICO); g.bind("org", ORG); g.bind("schema", SCHEMA)
    g.bind("nam1", NAM1); g.bind("nam2", NAM2); g.bind("nam3", NAM3)
    g.bind("nadat1", NADAT1); g.bind("nadat2", NADAT2); g.bind("nadat3", NADAT3)

    # 기존 인물 색인 + 임시 인물 넘버링
    idx_exact, idx_nospace = build_person_index_from_ttl(people_ttl_url) if people_ttl_url else ({}, {})
    print(f"[info] loaded persons: exact={len(idx_exact)}, nospace={len(idx_nospace)}")
    last_tmp_num = get_last_provisional_number(people_ttl_url)
    prov_mgr = ProvisionalPersonManager(existing_max_num=last_tmp_num)

    # 기존 d1 식별자 선점
    existing_ids = load_existing_identifiers(existing_d1_ttl)
    print(f"[info] existing d1 identifiers loaded: {len(existing_ids)}")

    # d1의 rico:Date 선점 및 중복 진단
    date_map = load_existing_date_ids(existing_d1_ttl)
    pre_seen_date = set(date_map.keys())
    dup = {k: v for k, v in date_map.items() if len(set(v)) > 1}
    if dup:
        print("[warn] rico:Date identifier 중복 발생(동일 식별자, 서로 다른 주체):")
        for k, vals in dup.items():
            print(f"  - {k}: {sorted(set(vals))}")
    else:
        print("[info] rico:Date identifier 중복 없음(마스터+imports 기준).")

    # 회기 시작연도 파악용 master 그래프
    g_master_for_dates = parse_with_imports(existing_d1_ttl) if existing_d1_ttl else Graph()

    # seen 세트
    seen = {
        "event": set(), "activity": set(),
        "rs_root": set(), "rs_branch": set(), "rs_session": set(),
        "record": set(), "inst": set(), "date": set(), "daterange": set()
    }
    for date_id in pre_seen_date:
        seen["date"].add(date_id)

    def preclaim(identifier: str, bucket: str):
        if identifier in existing_ids:
            seen[bucket].add(identifier)

    event_participants: Dict[str, set] = {}
    session_info: Dict[int, Dict[str, object]] = {}
    record_session: Dict[str, int] = {}

    total_rows = 0
    linked_existing = 0
    linked_provisional = 0

    # ===== 집계용 카운터(의원ID 있는 행만 대상) =====
    member_total = 0
    member_match_by_code = 0
    member_match_by_name = 0
    member_prov_rows = 0

    for _, row in df.iterrows():
        total_rows += 1

        mtg_no = int(row["회의번호"]) if not pd.isna(row["회의번호"]) else None
        if mtg_no is None:
            continue

        generation = int(row["대수"]) if not pd.isna(row["대수"]) else None
        session_raw = row.get("회수")
        round_raw   = row.get("차수")
        date_raw    = row.get("회의일자")

        session_num = digits_only(session_raw)
        session4    = pad(session_num, 4)
        round_num   = digits_only(round_raw)
        round3      = pad(round_num, 3)

        # ===== Event =====
        ev_id = f"EV-M-{mtg_no}"
        preclaim(ev_id, "event")
        if ev_id not in seen["event"]:
            ev = URIRef(NADAT1[ev_id])
            seen["event"].add(ev_id)
            g.add((ev, RDF.type, RICO.Event))
            add_lit(g, ev, RICO.identifier, ev_id)

            lab = f"[{ev_id}]"
            if session_num is not None: lab += f" 제{session_num}회"
            if round_num  is not None:  lab += f" 제{round_num}차"
            lab += f" {COMMITTEE_LABEL_KO}"
            add_lit(g, ev, SKOS.prefLabel, lab, lang="ko")
            add_lit(g, ev, RDFS.label,     lab, lang="ko")

            name_txt = ""
            if generation is not None: name_txt += f"제{generation}대국회 "
            if session_num is not None: name_txt += f"제{session_num}회 "
            if round_num  is not None: name_txt += f"제{round_num}차 "
            name_txt += f"{COMMITTEE_LABEL_KO}"
            add_lit(g, ev, RICO.name, name_txt.strip(), lang="ko")
            
            add_res(g, ev, RICO.hasEventType, str(NAM1["StandingCommitteeMeeting"]))
            event_participants[ev_id] = set()
        ev = URIRef(NADAT1[ev_id])

        # ===== Activity (상임위 심의) =====
        act_id = f"ACT-M-{mtg_no}-002"
        preclaim(act_id, "activity")
        if act_id not in seen["activity"]:
            act = URIRef(NADAT1[act_id])
            seen["activity"].add(act_id)
            g.add((act, RDF.type, RICO.Activity))
            add_lit(g, act, RICO.identifier, act_id)

            act_lab = f"[{act_id}]"
            if session_num is not None: act_lab += f" 제{session_num}회"
            if round_num  is not None: act_lab += f" 제{round_num}차"
            act_lab += f" {COMMITTEE_LABEL_KO} 심의"
            add_lit(g, act, SKOS.prefLabel, act_lab, lang="ko")
            add_lit(g, act, RDFS.label,     act_lab, lang="ko")
            add_lit(g, act, RICO.name, act_lab.split("]", 1)[-1].strip(), lang="ko")

            add_res(g, act, RICO.hasActivityType, str(NAM1["CommitteeReviewAction"]))
            add_res(g, act, RICO.isOrWasPartOf, str(NADAT1[ev_id]))
            add_res(g, act, RICO.isOrWasPerformedBy, COMMITTEE_IRI)
        act = URIRef(NADAT1[act_id])

        # ===== RecordSet 계층 =====
        if generation is not None:
            rs_root_id = f"RS1000{generation}"
            rs_branch_id = f"RS1000{generation}-{BRANCH}"
            preclaim(rs_branch_id, "rs_branch")
            if rs_branch_id not in seen["rs_branch"]:
                rs_branch = URIRef(NADAT1[rs_branch_id])
                seen["rs_branch"].add(rs_branch_id)
                g.add((rs_branch, RDF.type, RICO.RecordSet))
                add_lit(g, rs_branch, RICO.identifier, rs_branch_id)
                add_lit(g, rs_branch, SKOS.prefLabel, f"[{rs_branch_id}] {generation}대 {COMMITTEE_LABEL_KO} 회의록", lang="ko")
                add_lit(g, rs_branch, RDFS.label,     f"[{rs_branch_id}] {generation}대 {COMMITTEE_LABEL_KO} 회의록", lang="ko")
                add_lit(g, rs_branch, RICO.classification, "A")
                add_lit(g, rs_branch, RICO.name, f"제{generation}대국회 {COMMITTEE_LABEL_KO} 회의록", lang="ko")
                g.add((URIRef(NADAT1[rs_root_id]), RICO.includesOrIncluded, rs_branch))

        if generation is not None and session4 is not None:
            rs_session_id = f"RS1000{generation}-{BRANCH}-{session4}"
            preclaim(rs_session_id, "rs_session")
            if rs_session_id not in seen["rs_session"]:
                rs_session = URIRef(NADAT1[rs_session_id])
                seen["rs_session"].add(rs_session_id)
                g.add((rs_session, RDF.type, RICO.RecordSet))
                add_lit(g, rs_session, RICO.identifier, rs_session_id)
                lab = f"[{rs_session_id}] {generation}대 {session_num}회 {COMMITTEE_LABEL_KO} 회의록" if session_num is not None else f"[{rs_session_id}] {generation}대 {COMMITTEE_LABEL_KO} 회의록"
                add_lit(g, rs_session, SKOS.prefLabel, lab, lang="ko")
                add_lit(g, rs_session, RDFS.label,     lab, lang="ko")  
                add_lit(g, rs_session, RICO.classification, "A")
                if session_num is not None:
                    add_lit(g, rs_session, RICO.name, f"제{generation}대국회 제{session_num}회 {COMMITTEE_LABEL_KO} 회의록", lang="ko")
                else:
                    add_lit(g, rs_session, RICO.name, f"제{generation}대국회 {COMMITTEE_LABEL_KO} 회의록", lang="ko")
                g.add((URIRef(NADAT1[f"RS1000{generation}-{BRANCH}"]), RICO.includesOrIncluded, rs_session))

                if session_num is not None:
                    g.add((rs_session, RICO.hasOrHadAllMembersWithCreationDate,
                           URIRef(NADAT1[f"date-session-{session_num}"])))

        # ===== Record =====
        if generation is not None and session4 is not None and round3 is not None:
            r_id = f"R1000{generation}{session4}{round3}{mtg_no}"
            preclaim(r_id, "record")
            if r_id not in seen["record"]:
                seen["record"].add(r_id)
                r = URIRef(NADAT1[r_id])
                g.add((r, RDF.type, RICO.Record))
                add_lit(g, r, RICO.identifier, r_id)

                rn = digits_only(round_raw)
                rn_str = f"{rn}차" if rn is not None else ""
                session_part = f"{session_num}회 " if session_num is not None else ""
                r_label = f"[{r_id}] {generation}대 {session_part}{rn_str} {COMMITTEE_LABEL_KO} 회의록"
                add_lit(g, r, SKOS.prefLabel, r_label.strip(), lang="ko")
                add_lit(g, r, RDFS.label,     r_label.strip(), lang="ko")  

                title = ""
                if session_num is not None: title += f"제{session_num}회국회 "
                title += f"{COMMITTEE_LABEL_KO}회의록 "
                if rn is not None: title += f"제 {rn} 호"
                add_lit(g, r, RICO.title, title.strip(), lang="ko")

                add_res(g, r, RICO.hasDocumentaryFormType, str(NAM1["Minutes"]))
                add_res(g, r, RICO.hasContentOfType, str(NAM1["text"]))
                add_res(g, r, RICO.hasOrHadLanguage, str(NAM1["lang-kor"]))
                g.add((r, RICO.documents, act))
                add_res(g, r, RICO.hasCreator, str(NADAT2["NAS-0001-0001"]))
                g.add((URIRef(NADAT1[f"RS1000{generation}-{BRANCH}-{session4}"]), RICO.includesOrIncluded, r))

                ymd = parse_date_simple(date_raw)
                if ymd:
                    y, m, d = ymd
                    d_node = ensure_single_date_node(g, seen, y, m, d)
                    g.add((d_node, RICO.isDateOfOccurrenceOf, URIRef(NADAT1[ev_id])))
                    g.add((r, RICO.hasCreationDate, d_node))
                    if session_num is not None:
                        g.add((d_node, RICO.isWithin, URIRef(NADAT1[f"date-session-{session_num}"])))

                    if session_num is not None:
                        info = session_info.get(session_num, {"min": None, "max": None, "records": []})
                        cur = (y, m, d)
                        if info["min"] is None or cur < info["min"]:
                            info["min"] = cur
                        if info["max"] is None or cur > info["max"]:
                            info["max"] = cur
                        info["records"].append(r_id)
                        session_info[session_num] = info
                        record_session[r_id] = session_num

                # ===== Instantiation (전자회의록 -02) =====
                gen2 = f"{generation:02d}" if generation is not None else "00"
                base_key = f"{gen2}{session4}{round3}{mtg_no}"
                ins2_id  = f"INS-{BRANCH}-{base_key}-02"
                preclaim(ins2_id, "inst")
                if ins2_id not in seen["inst"]:
                    seen["inst"].add(ins2_id)
                    ins2 = URIRef(NADAT1[ins2_id])
                    g.add((ins2, RDF.type, RICO.Instantiation))
                    add_lit(g, ins2, RICO.identifier, ins2_id)
                    add_lit(g, ins2, SKOS.prefLabel, f"[{ins2_id}] {COMMITTEE_LABEL_KO} 전자회의록", lang="ko")
                    add_lit(g, ins2, RDFS.label,     f"[{ins2_id}] {COMMITTEE_LABEL_KO} 전자회의록", lang="ko")  
                    add_lit(g, ins2, RICO.title, title.strip(), lang="ko")
                    add_lit(g, ins2, RICO.conditionsOfAccess, "공개(대국민 열람 가능)", lang="ko")
                    add_lit(
                        g, ins2, RICO.conditionsOfUse,
                        "공공누리 제3유형 (출처표시 + 변경금지) 조건에 따라 자유롭게 이용이 가능합니다.", lang="ko"
                    )
                    add_res(g, ins2, RICO.hasOrHadManager, str(NADAT2["NAS-0002-0001"]))
                    g.add((r, RICO.hasOrHadInstantiation, ins2))

                    add_res(g, ins2, RICO.hasCarrierType,              str(NAM1["DigitalFile"]))
                    add_res(g, ins2, RICO.hasRepresentationType,       str(NAM1["Text"]))
                    add_res(g, ins2, RICO.hasProductionTechniqueType,  str(NAM1["DigitalCreation"]))

        # ===== Participants =====
        speaker_raw = row.get("발언자")
        if speaker_raw is None or (isinstance(speaker_raw, float) and pd.isna(speaker_raw)) or str(speaker_raw).strip() == "":
            continue

        name, iri, role, original = parse_speaker_name(speaker_raw, idx_exact, idx_nospace)
        found_by_name = bool(iri)
        found_by_code = False

        # 식별자 " / PER-.." 우선
        code_from_speaker = extract_person_code_from_speaker(original)
        if code_from_speaker:
            iri = str(NADAT2[code_from_speaker])
            found_by_code = True
            found_by_name = False

        # 의원ID 유무만 확인(고유키로 쓰지 않음)
        member_id_raw = row.get("의원ID")
        has_member_id = (member_id_raw is not None) and (str(member_id_raw).strip() != "") and (str(member_id_raw).lower() != "nan")

        # === d2 매칭 성공/실패에 따른 그래프 추가 ===
        if iri:
            if iri not in event_participants.setdefault(ev_id, set()):
                event_participants[ev_id].add(iri)
                add_res(g, ev, RICO.hasOrHadParticipant, iri)
                linked_existing += 1
        else:
            # d2 매칭 실패 → 임시개체
            speaker_text = original if original else speaker_raw
            tmp_iri = prov_mgr.get_or_create(
                speaker_text,
                member_id=str(member_id_raw).strip() if has_member_id else None,
                role_text=role
            )
            if tmp_iri and tmp_iri not in event_participants.setdefault(ev_id, set()):
                event_participants[ev_id].add(tmp_iri)
                add_res(g, ev, RICO.hasOrHadParticipant, tmp_iri)
                linked_provisional += 1

        # === 집계는 '의원ID가 있는 행'만 ===
        if has_member_id:
            member_total += 1
            if iri:
                if found_by_code:
                    member_match_by_code += 1
                elif found_by_name:
                    member_match_by_name += 1
                else:
                    member_match_by_name += 1
            else:
                member_prov_rows += 1

    # ===== 연도별(회기 시작연도 기준) 관련 개체 생성 =====
    year_to_sessions: DefaultDict[int, List[int]] = defaultdict(list)
    for sess, info in session_info.items():
        start_y_from_session = parse_session_start_year_from_normalized(g_master_for_dates, sess)
        if start_y_from_session is not None:
            year_to_sessions[start_y_from_session].append(sess)
        elif info["min"]:
            year_to_sessions[info["min"][0]].append(sess)

    for start_year, sessions in sorted(year_to_sessions.items()):
        min_ymd = None
        max_ymd = None
        grouped_records: List[str] = []
        for sess in sessions:
            info = session_info[sess]
            if info["min"]:
                if (min_ymd is None) or (info["min"] < min_ymd):
                    min_ymd = info["min"]
            if info["max"]:
                if (max_ymd is None) or (info["max"] > max_ymd):
                    max_ymd = info["max"]
            grouped_records.extend(info["records"])

        if (min_ymd is None) or (max_ymd is None):
            continue

        sy, sm, sd = min_ymd
        ey, em, ed = max_ymd

        start_node = ensure_single_date_node(g, seen, sy, sm, sd)
        end_node   = ensure_single_date_node(g, seen, ey, em, ed)

        def recid_to_mtg_no(rid: str) -> Optional[int]:
            m = re.search(r"R1000\d{2}\d{4}\d{3}(\d+)$", rid)
            return int(m.group(1)) if m else None

        mtg_nos = [recid_to_mtg_no(r) for r in grouped_records]
        mtg_nos = [n for n in mtg_nos if n is not None]
        if not mtg_nos:
            continue

        m_start, m_end = min(mtg_nos), max(mtg_nos)

        # === 연도별 보존집합 RecordSet 생성 === 
        year_key = start_year  # 시작연 기준(연도)
        rs_year_id = f"RS1000{generation}{'-'+BRANCH if BRANCH else ''}-Y{year_key}"
    
        # 연도별 통일 명명
        year_name_ko = f"제{generation}대국회 {COMMITTEE_LABEL_KO} 회의록 ({year_key})"
    
        if rs_year_id not in seen["rs_session"]:
            seen["rs_session"].add(rs_year_id)
            rs_year = URIRef(NADAT1[rs_year_id])
    
            g.add((rs_year, RDF.type, RICO.RecordSet))
            add_lit(g, rs_year, RICO.identifier, rs_year_id)
            add_lit(g, rs_year, SKOS.prefLabel, f"[{rs_year_id}] {year_name_ko}", lang="ko")
            add_lit(g, rs_year, RDFS.label,     f"[{rs_year_id}] {year_name_ko}", lang="ko")
            add_lit(g, rs_year, RICO.classification, "A")
            add_lit(g, rs_year, RICO.name, year_name_ko, lang="ko")
            add_lit(g, rs_year, DCTERMS.coverage, f"{ymd_to_iso(sy, sm, sd)}/{ymd_to_iso(ey, em, ed)}")
    
            parent_branch_rs = URIRef(NADAT1[f"RS1000{generation}-{BRANCH}"])
            g.add((parent_branch_rs, RICO.includesOrIncluded, rs_year))
        else:
            rs_year = URIRef(NADAT1[rs_year_id])
    
        # 해당 연도의 모든 record들을 RS에 편입
        for r_local in grouped_records:
            r_ref = URIRef(NADAT1[r_local])
            g.add((rs_year, RICO.includesOrIncluded, r_ref))

        # === 보존본 Instantiation(-01) ===
        ins1_id = f"INS-{BRANCH}-{m_start}_{m_end}-01"
        preclaim(ins1_id, "inst")
        if ins1_id not in seen["inst"]:
            seen["inst"].add(ins1_id)
            ins1 = URIRef(NADAT1[ins1_id])
            g.add((ins1, RDF.type, RICO.Instantiation))
            add_lit(g, ins1, RICO.identifier, ins1_id)

        ins1_name = f"제{generation}대국회 {COMMITTEE_LABEL_KO} 보존회의록 ({year_key})"
        full_pref = f"[{ins1_id}] {ins1_name}"
        add_lit(g, ins1, SKOS.prefLabel, full_pref, lang="ko")
        add_lit(g, ins1, RDFS.label,     full_pref, lang="ko")  
        add_lit(g, ins1, RICO.title, ins1_name, lang="ko")

        # ---- 보존본(-01) 매체/조건/주석 ----
        add_res(g, ins1, RICO.hasCarrierType,             str(NAM1["Paper"]))
        add_res(g, ins1, RICO.hasRepresentationType,      str(NAM1["Text"]))
        add_res(g, ins1, RICO.hasProductionTechniqueType, str(NAM1["Printing"]))
        add_lit(g, ins1, RICO.conditionsOfAccess, "제한(현직 국회의원만 열람 가능)", lang="ko")
        add_lit(g, ins1, RDFS.comment, "권수 미상 (임시 생성)", lang="ko")
        add_res(g, ins1, RICO.hasOrHadHolder,  str(NADAT2["NAArchives"]))
        add_res(g, ins1, RICO.hasOrHadManager, str(NADAT2["NAArchives"]))
        add_res(g, ins1, RICO.hasCreator,      str(NADAT2["NAS-0001-0001"]))

        g.add((ins1, RICO.hasBeginningDate, start_node))
        g.add((ins1, RICO.hasEndDate,       end_node))
        add_lit(g, ins1, DCTERMS.coverage, f"{ymd_to_iso(sy, sm, sd)}/{ymd_to_iso(ey, em, ed)}")

        for r_local in grouped_records:
            g.add((URIRef(NADAT1[r_local]), RICO.hasOrHadInstantiation, URIRef(NADAT1[ins1_id])))

        print(f"[info] archival bundle created: {ins1_id} (sessions={sorted(sessions)}, records={len(grouped_records)})")

    out_file = Path(resolved_out_ttl).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=out_file.as_posix(), format="turtle")

    # ===== 요약 로그 =====
    def _ratio(n, d):
        return (n / d * 100.0) if d else 0.0

    print(f"[info] provisional created (unique) = {len(prov_mgr.entries)}")
    print(f"[info] member-ID rows: total={member_total}, "
          f"matched(by code)={member_match_by_code}, matched(by name)={member_match_by_name}, "
          f"provisional(rows)={member_prov_rows}, "
          f"link_rate={_ratio(member_match_by_code + member_match_by_name, member_total):.2f}%")

    print(f"[write] {out_file}")

    prov_mgr.write_ttl_txt(resolved_out_tmp_people)

# ==== 실행 ====
if __name__ == "__main__":
    convert(
        CSV_PATH,
        OUT_TTL,
        OUT_TMP_PEOPLE,
        encoding=CSV_ENCODING,
        people_ttl_url=PEOPLE_TTL,
        existing_d1_ttl=EXISTING_D1_TTL
    )
