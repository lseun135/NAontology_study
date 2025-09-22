
<br/>
[https://lseun135.github.io/NAontology_study/](https://lseun135.github.io/NAontology_study/)
<br/><br/>

### RiC 요소 정리
ICA-EGAD에서 제공한 [RiC-O components csv](https://github.com/ICA-EGAD/RiC-O/tree/master/ontology/current-version/CSV_lists_of_components)를 활용하여 정리 
<br/>(Notion 페이지로 이동)
* [RiC-CM](https://www.notion.so/1784183a63dc807fbd58ecfdafd1225d?v=1784183a63dc8044ba19000c0911e211&source=copy_link)
* [RiC-O](https://www.notion.so/1874183a63dc803a9f85e81012cf32cc?v=1874183a63dc80ceb73c000c1c3457ca&source=copy_link)
<br/><br/>

# 구축 상황
## T box
* [m1-record](https://github.com/lseun135/NAontology_study/blob/main/docs/m1-record.ttl)
* [m2-person](https://github.com/lseun135/NAontology_study/blob/main/docs/m2-person.ttl)
* [m3-content](https://github.com/lseun135/NAontology_study/blob/main/docs/m3-content.ttl)
<br/>

## A box
현재 용량이 부족하므로 파일을 분리해서 게시할 예정 (모듈과 데이터 구조는 유지)<br/>
→소관위원회 또는 개체별 분리를 고려하고 있음
* [d1-record](https://github.com/lseun135/NAontology_study/blob/main/docs/resource/d1-record.ttl)
* [d2-person](https://github.com/lseun135/NAontology_study/blob/main/docs/resource/d2-person.ttl)
* [d3-content](https://github.com/lseun135/NAontology_study/blob/main/docs/resource/d3-content.ttl)
<br/>

## URL import시, 유의사항
URL 뒤에 '.ttl'을 붙일 것 (오류방지)
* 예시: https://lseun135<k>.github.io/NAontology_study/resource/d1-record **.ttl**
  
<br/><br/>
# NameSpace
* 자체 온톨로지
  | prefix | IRI |
  |-|-|
  | nam1 | https://lseun135.github.io/NAontology_study/m1-record# |
  | nam2 | https://lseun135.github.io/NAontology_study/m2-person# |
  | nam3 | https://lseun135.github.io/NAontology_study/m3-content# |
  | nadat1 | https://lseun135.github.io/NAontology_study/resource/d1-record# |
  | nadat2 | https://lseun135.github.io/NAontology_study/resource/d2-person# |
  | nadat3 | https://lseun135.github.io/NAontology_study/resource/d3-content# |
<br/>

* 공통 온톨로지
  | prefix | IRI |
  |-|-|
  | owl | http://www.w3.org/2002/07/owl# |
  | rdf | http://www.w3.org/1999/02/22-rdf-syntax-ns# |
  | rdfs | http://www.w3.org/2000/01/rdf-schema# |
  | xml | http://www.w3.org/XML/1998/namespace |
  | xsd | http://www.w3.org/2001/XMLSchema# |
  | dcterms | http://purl.org/dc/terms/ |
  | schema | https://schema.org/ |
  | skos | https://www.w3.org/2004/02/skos/core# |
  | prov | http://www.w3.org/ns/prov# |
<br/>

* 주요 온톨로지
  | prefix | IRI |
  |-|-|
  | rico | https://www.ica.org/standards/RiC/ontology# |
  | opengov | http://www.w3.org/ns/opengov# |
  | crm | http://www.cidoc-crm.org/cidoc-crm/ |
<br/>

* 기타 온톨로지
  | prefix | IRI |
  |-|-|
  | bf | https://id.loc.gov/ontologies/bibframe/ |
  | org | http://www.w3.org/ns/org# |
  | foaf | http://xmlns.com/foaf/0.1/ |
  | bio | http://purl.org/vocab/bio/0.1/ |
  | gn | https://www.geonames.org/ontology# |
  | adms | http://www.w3.org/ns/adms# |

