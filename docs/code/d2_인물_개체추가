import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import re
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, DCTERMS, SKOS, OWL, FOAF, XSD

# ================= 사용자 환경 =================
SERVICE_KEY = ""
INPUT_PATH  = "D:/연구/국회회의록/251101_d2_C.txt"     # 기존 ABox 파일
OUTPUT_PATH = "D:/연구/국회회의록/251101_d2_C_P.txt"   # 결과 저장 파일

# ================= API =================
API_A = "https://open.assembly.go.kr/portal/openapi/nwvrqwxyaytdsfvhu"   # 현황(22대 중심, 보강용)
API_B = "https://open.assembly.go.kr/portal/openapi/ALLNAMEMBER"         # 역대(21대 포함, 기준 데이터)

# ================= Prefix / NS =================
ORG      = Namespace("http://www.w3.org/ns/org#")
DCTERMS  = Namespace("http://purl.org/dc/terms/")
GN       = Namespace("https://www.geonames.org/ontology#")
SCHEMA   = Namespace("https://schema.org/")
BIO      = Namespace("http://purl.org/vocab/bio/0.1/")
DAT2     = Namespace("https://lseun135.github.io/NAontology_study/resource/d2-person#")
NAM2     = Namespace("https://example.org/NAontology/m2-person#")

NA      = DAT2["NationalAssembly"]   # 국회
RL_PA   = DAT2["RL-PA"]              # 정당소속 역할(공통)

# ================= 유틸/상수 =================
_ASCII_CODE = re.compile(r"^[A-Za-z0-9_-]+$")

# ---- 화이트리스트 (라벨@ko → (LocalName, ASCII 약자)) ----
PARTY_ALLOW = {
    "기본소득당":       ("BasicIncomeParty",          "20BIP"),
    "더불어민주당":     ("DemocraticPartyofKorea",    "14DPK"),
    "더불어시민당":      ("PlatformParty",            "20PFP"),
    "국민의힘":         ("PeoplePowerParty",          "20PPP"),
    "미래통합당":       ("UnitedFutureParty",         "20UFP"),
    "미래한국당":       ("FutureKoreaParty",          "20FKP"),
    "진보당":           ("ProgressiveParty",          "17PGP"),
    "정의당":           ("JusticeParty",              "12JTP"),
    "조국혁신당":       ("RebuildingKoreaParty",      "24RKP"),
    "개혁신당":         ("ReformParty",               "24RFP"),
    "국민의당":         ("PeoplesParty",              "20PSP"),
    "열린민주당":       ("OpenDemocraticParty",       "20ODP"),
}

def load_graph(path: str) -> Graph:
    g = Graph()
    for fmt in ["turtle","xml","n3","nt","trig","json-ld"]:
        try:
            g.parse(path, format=fmt); break
        except Exception:
            continue
    g.bind("org", ORG); g.bind("skos", SKOS); g.bind("dcterms", DCTERMS); g.bind("foaf", FOAF)
    g.bind("nadat2", DAT2); g.bind("nam2", NAM2); g.bind("gn", GN); g.bind("schema", SCHEMA); g.bind("bio", BIO)
    g.bind("owl", OWL)
    return g

def fetch_rows_xml(api_url, pindex=1, psize=100):
    params = {"Key": SERVICE_KEY, "Type": "xml", "pIndex": pindex, "pSize": psize}
    url = f"{api_url}?{urlencode(params)}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    root = ET.fromstring(r.text)
    rows = []
    for row in root.findall(".//row"):
        record = {child.tag: (child.text or "").strip() for child in row}
        rows.append(record)
    return rows

def normalize_gender(val: str) -> str | None:
    if val in ("남","M","male","Male"): return "male"
    if val in ("여","F","female","Female"): return "female"
    return None

def calendar_from_A(val: str) -> str | None:
    return "lunar" if val == "음" else ("solar" if val == "양" else None)

def calendar_from_B(val: str) -> str | None:
    return "solar" if val == "양" else ("lunar" if val == "음" else None)

def first_nonempty(*vals):
    for v in vals:
        if v: return v
    return None

def parse_term_token(tok: str) -> int | None:
    m = re.search(r"제\s*(\d+)\s*대", (tok or "").strip())
    return int(m.group(1)) if m else None

def find_committee_node(g: Graph, label_ko: str) -> URIRef | None:
    if not label_ko: return None
    for s in g.subjects(SKOS.prefLabel, Literal(label_ko, lang="ko")):
        return s
    return None

def ensure_leg_session(g: Graph, term_no: int) -> URIRef:
    node = DAT2[f"LegSession-{term_no}"]
    if not list(g.predicate_objects(node)):
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, RDFS.label, Literal(f"제{term_no}대 입법회기", lang="ko")))
    return node

def ensure_legislator_role(g: Graph, term_no: int) -> URIRef:
    node = DAT2[f"RL-LM{term_no}"]
    if not list(g.predicate_objects(node)):
        g.add((node, RDF.type, ORG.Role))
        g.add((node, SKOS.prefLabel, Literal(f"제{term_no}대 국회의원", lang="ko")))
        g.add((node, SKOS.prefLabel, Literal(f"Member of the {term_no}th National Assembly", lang="en")))
    return node

def index_A_for_enrichment():
    idx = {}
    p = 1
    while True:
        rows = fetch_rows_xml(API_A, pindex=p, psize=100)
        if not rows: break
        for r in rows:
            nm  = r.get("HG_NM","")
            bth = r.get("BTH_DATE","")
            sex = r.get("SEX_GBN_NM","")
            if nm:
                idx[(nm, bth, sex)] = r
        if len(rows) < 100: break
        p += 1
    return idx

# ================= 핵심: B 기준 + A 보강 =================
def add_from_B_with_A_enrichment(g: Graph, brow: dict, a_index: dict):
    name_ko = brow.get("NAAS_NM")
    if not name_ko:
        return

    gen = normalize_gender(brow.get("NTR_DIV",""))

    a_candidates = []
    b_key = (name_ko, brow.get("BIRDY_DT",""), "남" if gen=="male" else ("여" if gen=="female" else ""))
    if b_key in a_index:
        a_candidates.append(a_index[b_key])
    if not a_candidates:
        for (nm,_,_), row in a_index.items():
            if nm == name_ko:
                a_candidates.append(row); break

    mona_cd = a_candidates[0].get("MONA_CD") if a_candidates else None
    naas_cd = brow.get("NAAS_CD")
    primary_code = first_nonempty(naas_cd, mona_cd)

    if not primary_code:
        return

    person = DAT2[f"PER-LM-{primary_code}"]

    g.remove((person, RDF.type, FOAF.Person))
    g.add((person, RDF.type, NAM2.Legislator))

    # ===== identifier 정리 (접두사 없음) =====
    ids = []
    if naas_cd and naas_cd.strip():
        ids.append(naas_cd.strip())
    if mona_cd and mona_cd.strip():
        ids.append(mona_cd.strip())
    for code in dict.fromkeys(ids):
        g.add((person, DCTERMS.identifier, Literal(code)))

    # ===== 이름 =====
    g.add((person, FOAF.name, Literal(name_ko, lang="ko")))
    g.add((person, RDFS.label, Literal(name_ko, lang="ko")))
    if brow.get("NAAS_CH_NM"):
        g.add((person, FOAF.name, Literal(brow["NAAS_CH_NM"], lang="ko-Hani")))
    if brow.get("NAAS_EN_NM"):
        g.add((person, FOAF.name, Literal(brow["NAAS_EN_NM"], lang="en")))

    # ===== 성별 / 생일 =====
    if gen:
        g.set((person, FOAF.gender, Literal(gen)))

    cal_b = calendar_from_B(brow.get("BIRDY_DIV_CD",""))
    bday  = brow.get("BIRDY_DT","")
    if cal_b == "solar" and bday:
        g.add((person, NAM2.calendarType, Literal("solar", datatype=XSD.token)))
        g.add((person, SCHEMA.birthDate, Literal(bday, datatype=XSD.date)))
    elif cal_b == "lunar" and bday:
        g.add((person, NAM2.calendarType, Literal("lunar", datatype=XSD.token)))
        g.add((person, NAM2.lunarBirthText, Literal(bday, datatype=XSD.date)))

    # ===== 이미지 =====
    url = brow.get("NAAS_PIC")
    if url:
        if url.startswith("http"):
            g.add((person, SCHEMA.image, URIRef(url)))
        else:
            g.add((person, SCHEMA.image, Literal(url, datatype=XSD.anyURI)))

    # ===== 연락처 및 홈페이지 =====
    if brow.get("NAAS_EMAIL_ADDR"):
        g.add((person, SCHEMA.email, Literal(brow["NAAS_EMAIL_ADDR"])))
    if brow.get("NAAS_TEL_NO"):
        g.add((person, SCHEMA.telephone, Literal(brow["NAAS_TEL_NO"])))
    if brow.get("NAAS_HP_URL"):
        g.add((person, SCHEMA.url, URIRef(brow["NAAS_HP_URL"])))
    if brow.get("BRF_HST"):
        g.add((person, BIO.biography, Literal(brow["BRF_HST"])))

    # ==== 재선/대수 설명 ====
    rlct = (brow.get("RLCT_DIV_NM") or "").strip()
    gter = (brow.get("GTELT_ERACO") or "").strip()
    if rlct and gter:
        g.set((person, DCTERMS.description, Literal(f"{rlct}({gter})", lang="ko")))
    elif rlct:
        g.set((person, DCTERMS.description, Literal(rlct, lang="ko")))
    elif gter:
        g.set((person, DCTERMS.description, Literal(gter, lang="ko")))

    # ===== A API 보강(비어있을 때만) — homepage → schema:url 로 변경 =====
    if a_candidates:
        arow = a_candidates[0]
        if not list(g.objects(person, SCHEMA.telephone)) and arow.get("TEL_NO"):
            g.add((person, SCHEMA.telephone, Literal(arow["TEL_NO"])))
        if not list(g.objects(person, SCHEMA.email)) and arow.get("E_MAIL"):
            g.add((person, SCHEMA.email, Literal(arow["E_MAIL"])))
        if not list(g.objects(person, SCHEMA.url)) and arow.get("HOMEPAGE"):
            g.add((person, SCHEMA.url, URIRef(arow["HOMEPAGE"])))

    # ==== 대수/선거구/위원회/보좌진 등 ====
    eras_raw   = brow.get("GTELT_ERACO","")
    era_tokens = [t.strip() for t in eras_raw.split(",") if t.strip()]
    terms      = [t for t in (parse_term_token(t) for t in era_tokens) if t]

    latest_term = max(terms) if terms else None

    elec_parts  = [x.strip() for x in (brow.get("ELECD_NM", "") or "").split("/")]
    party_parts = [x.strip() for x in (brow.get("PLPT_NM", "")  or "").split("/")]
    cmit_parts  = [x.strip() for x in (brow.get("BLNG_CMIT_NM","") or "").split("/")]

    for idx, term_no in enumerate(terms):
        leg = ensure_leg_session(g, term_no)
        rl  = ensure_legislator_role(g, term_no)

        mem_code = f"1000{int(term_no):02d}"
        mem_na = DAT2[f"MEM-{primary_code}-{mem_code}"]

        g.add((mem_na, RDF.type, ORG.Membership))
        g.add((mem_na, ORG.member, person))
        g.add((mem_na, ORG.organization, NA))
        g.add((mem_na, ORG.role, rl))
        g.add((mem_na, DCTERMS.temporal, leg))
        g.add((mem_na, RDFS.label, Literal(f"{name_ko}: 제{term_no}대 국회 membership", lang="ko")))

        elec_label = (elec_parts[idx] if 0 <= idx < len(elec_parts)
                      else (elec_parts[-1] if elec_parts else ""))
        if elec_label:
            g.add((mem_na, DCTERMS.coverage, Literal(elec_label, lang="ko")))

        cmit_blob = (cmit_parts[idx] if 0 <= idx < len(cmit_parts)
                     else (cmit_parts[-1] if cmit_parts else ""))
        if cmit_blob:
            for label in [x.strip() for x in cmit_blob.split(",") if x.strip()]:
                node = find_committee_node(g, label)
                if node is not None:
                    mem_c = BNode()
                    g.add((mem_c, RDF.type, ORG.Membership))
                    g.add((mem_c, ORG.member, person))
                    g.add((mem_c, ORG.organization, node))
                    g.add((mem_c, DCTERMS.temporal, leg))
                    g.add((mem_c, RDFS.label,
                           Literal(f"{name_ko}: {label} 위원(제{term_no}대)", lang="ko")))

        if latest_term and term_no == latest_term:
            office_txt = None
            if brow.get("OFFM_RNUM_NO"):
                office_txt = f"서울특별시 영등포구 의사당대로 1, {brow['OFFM_RNUM_NO']}"
            elif a_candidates and a_candidates[0].get("ASSEM_ADDR"):
                office_txt = f"서울특별시 영등포구 의사당대로 1, {a_candidates[0]['ASSEM_ADDR']}"

            if office_txt:
                g.add((mem_na, SCHEMA.address, Literal(office_txt)))

            def _add_staff(csv_text, pred):
                if not csv_text: return
                for n in [x.strip() for x in csv_text.split(",") if x.strip()]:
                    g.add((mem_na, pred, Literal(n)))

            _add_staff(brow.get("AIDE_NM"),       NAM2.assistantName)
            _add_staff(brow.get("CHF_SCRT_NM"),   NAM2.seniorSecretaryName)
            _add_staff(brow.get("SCRT_NM"),       NAM2.secretaryName)

            if a_candidates:
                arow = a_candidates[0]
                _add_staff(arow.get("STAFF"),     NAM2.assistantName)
                _add_staff(arow.get("SECRETARY"), NAM2.seniorSecretaryName)
                _add_staff(arow.get("SECRETARY2"),NAM2.secretaryName)

    # ===================== 정당 멤버십 (화이트리스트 최신 1건) =====================
    latest_party_label = party_parts[-1] if party_parts else None
    if latest_party_label:
        info = PARTY_ALLOW.get(latest_party_label)
        if info:
            localname, abbr = info
            if _ASCII_CODE.fullmatch(abbr):
                party_node = DAT2[localname]
                mem_party  = DAT2[f"MEM-{primary_code}-{abbr}"]

                g.add((mem_party, RDF.type, ORG.Membership))
                g.add((mem_party, ORG.member, person))
                g.add((mem_party, ORG.organization, party_node))
                g.add((mem_party, ORG.role, RL_PA))
                g.add((mem_party, RDFS.label, Literal(f"{name_ko}: 정당 membership", lang="ko")))


# ================= 메인 =================
def main():
    g = load_graph(INPUT_PATH)
    a_index = index_A_for_enrichment()

    p = 1
    while True:
        rows = fetch_rows_xml(API_B, pindex=p, psize=100)
        if not rows: break
        for brow in rows:
            if any(x in brow.get("GTELT_ERACO","") for x in ("제19대", "제20대", "제21대")):
                add_from_B_with_A_enrichment(g, brow, a_index)
        if len(rows) < 100: break
        p += 1

    g.serialize(OUTPUT_PATH, format="turtle")
    print(f"저장 완료: {OUTPUT_PATH}")

    party_memberships = len(list(g.triples((None, ORG.role, RL_PA))))
    print(f"[통계] 정당 멤버십 개수: {party_memberships}")

if __name__ == "__main__":
    main()
