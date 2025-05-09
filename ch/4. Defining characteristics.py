import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
import ast
import requests
import json

warnings.filterwarnings('ignore')
tqdm.pandas()

# teach_columns 정의
teach_columns = {
    "TIME_CSMTC_SEG E.18-22시": "저녁 시간에 화장품을 구매하는 고객",
    'INCOME 01_중하계층': "저소득 고객",
    'INCOME 02_중상계층': "고소득 고객",
    'ONLINE_CSMTIC_RATE 상': "온라인으로 화장품을 많이 구매한 고객",
    'DRGSTR_RATE 상' : "올리브영에서 화장품을 구매하며 트렌디한 화장품 소비가 많은 고객",
    'TIME_SEG 02.오전' : "주요 소비가 오전 시간대로 이른 아침부터 하루를 시작하는 고객",
    "TIME_SEG 05.야간": "주요 소비가 야간 시간대인 고객",
    'TIME_CAFE_SEG B.06-10시': "아침 시간에 카페를 이용 빈도가 높은 고객",
    'TIME_CAFE_SEG B.10-14시': "점심 시간에 카페를 이용 빈도가 높은 고객",
    'TIME_CAFE_SEG C.14-18시': "오후 시간에 카페 이용 빈도가 높은 고객",
    'PER_ERADP 상' : "트렌드에 민감하고 얼리어답터 성향이 높은 고객",
    'LFSTG 01.미혼추정' : "자신의 삶을 사랑하며 다양한 취미 활동을 즐기고 새로운 경험을 추구하는 고객",
    'PER_DOM_TRV 상': "국내 여행에 관심이 많은 고객",
    'PER_FRN_TRV 상': "해외 여행에 관심이 많은 고객",
    'PER_ONLINE 상': "온라인 구매를 많이 하는 고객",
    'ONLINE_RATE 상': "온라인 구매가 많은 고객",
    'SKINCARE_CNT_6M 상' : "최근에 피부 관리에 집중한 고객",
    'SKINCARE_CNT_1Y 상' : "피부 관리에 관심이 높은 고객",
    'PER_PRC_SNSV 상': "가격에 민감한 고객",
    'PER_INDR_SPRTS 상' : "건강한 일상을 살기 위해 요가, 필라테스 등 건강한 루틴을 실천하는 고객",
    'PER_OUTDOOR 상' : "외부 활동을 통해 다양하고 활발한 취미활동을 하는 열정적인 고객",
    'PER_DRIVE 상': "운전을 많이 하는 고객",
    'PER_REMM 상': "명품을 많이 구매하는 고객",
    'PER_ONGAME 상': "온라인 게임을 많이 하는 고객",
    'PER_MDLV 상': "배달 식품을 자주 먹는 고객",
    'PER_BENEFIT 상': "혜택에 반응하는 고객",
    'PER_OTT 상' : "OTT를 자주 소비하여 미디어 컨텐츠, 다양한 장르와 플랫폼에 관심이 높은 고객",
    'JOB 02_회사원': "회사원 고객",
    "JOB 01_전문직": "전문직 고객",
    'PER_PRMM_FASHION 상': "명품 의류를 많이 구매하는 고객",
    'LFSTG 05.성인자녀추정': "성인 자녀를 둔 고객",
    'PER_CAFE 상': "카페를 많이 이용하는 고객",
    'PER_HLTHY_FOOD 상' : "건강한 일상의 루틴 실천을 위해 샐러드 등 건강한 식단을 주로 먹는 고객",
    'PER_ORGNIC_FOOD 상' : "유기농 제품을 많이 구매하는 고객",
    'NUTRI_CNT' : "뷰티 관련 건강 보조 식품을 많이 구매하는 고객",
    'NUTRI_AMT' : "뷰티 관련 건강 보조 식품을 많이 구매하는 고객"
}


# csv 불러오기
desc = pd.read_csv(r"C:\test6\description.csv")
desc = desc.dropna(subset=["BASICDESC"]).reset_index(drop=True)
desc = desc.drop_duplicates(subset=["PRDNM"]).reset_index(drop=True)
desc = dict(zip(desc["SAPPRDCD"], desc["BASICDESC"]))

review = pd.read_csv(r"C:\test6\item2.csv")
review = review.drop_duplicates(subset="RPRS_PRD_CD").reset_index(drop=True)
review = dict(zip(review["RPRS_PRD_CD"], review["RPRS_PRD_NM"]))

df = pd.read_csv(r"C:\test6\merged_data.csv", encoding="utf-8-sig", index_col="Unnamed: 0")
df["sequence"] = df["sequence"].apply(lambda x: ast.literal_eval(x))
df["names"] = df["names"].apply(lambda x: ast.literal_eval(x))
df["tfidf"] = df["tfidf"].apply(lambda x: ast.literal_eval(x))
df["tfidf_name"] = df["tfidf_name"].apply(lambda x: ast.literal_eval(x))
df["external_cluster"] = df["external_cluster"].apply(lambda x: ast.literal_eval(x))

def id_to_name(key):
    try:
        return review[int(float(key))]
    except:
        return key

names = []
for i in df["sequence"]:
    names.append([id_to_name(w) for w in i])
df["names"] = names

tfidf = []
for i in df["tfidf"]:
    tfidf.append([id_to_name(w) for w in i])
df["tfidf_name"] = tfidf

# 소구점 찾기 (internal only)
clu = df.groupby(["cluster", "external_cluster_num"])[["tfidf", "external_cluster"]].first()
clu.index.names = ["internal_cluster_num", "external_cluster_num"]
clu["cluster"] = range(len(clu["tfidf"]))

retrieve = {}
description = {}
for num in tqdm(df.index):
    temp1 = df["tfidf"].loc[num]
    temp2 = df["tfidf_name"].loc[num]
    for a, i in enumerate(temp1):
        retrieve[i] = str(temp2[a]).replace("IF.", "").replace("이니스프리", "").strip()
        try:
            description[retrieve[i]] = desc[i]
        except:
            pass

# Azure GPT Settings
GPT4V_ENDPOINT = "https://apeus-cst-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"
GPT4V_KEY = "fb73c5cdb51d4167b9eb41345d059bfe"

headers = {
    "Content-Type": "application/json;",
    "api-key": GPT4V_KEY,
}

def respond(payload):
    response = requests.post(GPT4V_ENDPOINT, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error {response.status_code}: {response.text}"

# Internal customer characteristics
result = []
internal_info = []

idx = clu.index.get_level_values("internal_cluster_num").unique()

for temp in tqdm(idx):

    system1 = "너는 화장품 기업 소속의 마케팅 매니저야."
    user1 = """
    내가 제공하는 고객 정보를 활용해서, 이 고객이 어떤 특징을 가지고 있는지 한 문장으로 정의하고, 어떤 정보를 보고 그렇게 정의했는지 그 번호를 알려줘.
    정보는 1개 - 3개 활용해 줘. 단순히 사용한 정보들을 나열하지 말고, 정보들을 유기적으로 결합해서 고객 특징을 만들어야 해.

    우리가 원하는 답변의 예시는 다음과 같아: '바쁜 일상 속에서도 건강한 일상을 살기 위해 이른 아침부터 자기관리와 건강한 루틴을 실천하는 고객. 1번, 3번 정보.'
    """
    assistant1 = "알겠습니다. 정보와 제품 특징을 제공해 주세요."

    internal = ""
    tp = list(clu["tfidf"].loc[temp])[0]
    internal_info.append(tp)
    for a, i in enumerate(tp):
        try:
            internal += f"{a}번 정보: 이 고객은 과거에 {retrieve[i]}를 구매한 적이 있어.\n제품 특징: {description[retrieve[i]]}\n"
        except:
            internal += f"{a}번 정보: 이 고객은 과거에 {retrieve[i]}를 구매한 적이 있어.\n"

    payload = {
        "messages": [
            {"role": "system", "content": system1},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": internal}
        ]
    }

    decision = True
    while decision:
        try:
            result.append(respond(payload))
            decision = False
        except Exception as e:
            print(f"Error in request: {e}")
            decision = False

regexp = re.compile("[0-9]+번 정보")
items = []
for a, i in enumerate(result):
    matches = regexp.findall(i)
    if matches:
        temp = matches[0].split("번")
        temp2 = internal_info[a]
        item = []
        for w in temp[:-1]:
            item.append(temp2[int(w[-1])])
        items.append(item)
    else:
        items.append([])

internal_df = pd.DataFrame({"internal_cluster": list(idx), "focus_item": items, "characteristics": result})
internal_df.to_csv(r"C:\test6\internal_characteristics.csv", encoding="utf-8-sig")

# External customer characteristics
result = []
external_info = []

for idx, temp in tqdm(enumerate(clu["tfidf"])):

    system1 = "너는 화장품 기업 소속의 마케팅 매니저야."
    user1 = """
    내가 제공하는 고객 정보를 활용해서, 이 고객이 어떤 특징을 가지고 있는지 한 문장으로 정의하고, 어떤 정보를 보고 그렇게 정의했는지 그 번호를 알려줘.
    정보는 1개 - 3개 활용할 수 있어. 단순히 사용한 정보들을 나열하지 말고, 정보들을 유기적으로 결합해서 고객 특징을 만들어야 해.
    우리가 원하는 답변의 예시는 다음과 같아:
    '바쁜 일상 속에서도 건강한 일상을 살기 위해 이른 아침부터 자기관리와 건강한 루틴을 실천하는 고객. 2번 정보.'
    '주로 온라인에 몰입되는 도파민 넘치는 생활습관을 가지고 있는 고객. 0번, 2번 정보.'
    '이른 아침부터 하루를 시작하지만, 늘 시간과 에너지가 부족해 외식이나 차려먹는 밥 대신 배달 음식을 시켜먹는 고객. 0번, 1번, 3번 정보'
    """
    assistant1 = "알겠습니다. 정보와 제품 특징을 제공해 주세요."

    external = ""
    tp = clu["external_cluster"].iloc[idx]
    external_info.append(tp)
    for a, i in enumerate(tp):
        external += f"{a}번 정보: 이 고객은 {teach_columns[i]}이야.\n"

    payload = {
        "messages": [
            {"role": "system", "content": system1},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": external}
        ]
    }

    decision = True
    counter = 0
    while decision:
        try:
            result.append(respond(payload))
            decision = False
        except Exception as e:
            print(f"Attempt {counter}: Error in request: {e}")
            counter += 1
            if counter > 5:
                decision = False

items = []
for a, i in enumerate(result):
    matches = regexp.findall(i)
    if matches:
        temp = matches[0].split("번")
        temp2 = external_info[a]
        item = []
        for w in temp[:-1]:
            item.append(temp2[int(w[-1])])
        items.append(item)
    else:
        items.append([])

print("Results processed.")
external_df = pd.DataFrame({"external_item": items, "characteristics": result}, index=clu.index)
external_df.to_csv(r"C:\test6\external_characteristics.csv", encoding="utf-8-sig")

print("Process completed.")