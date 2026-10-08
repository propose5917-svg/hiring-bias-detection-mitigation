"""Build counterfactual hiring-audit pairs from Resume.csv + postings.csv.

Each (resume, job) pair is rendered with a White-associated and a Black-associated
name (Bertrand & Mullainathan 2004), same gender, everything else identical.
Qualified = resume category matches the job's category; unqualified = job from a
distant category cluster.
"""
import os, re, random, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Paths to the two provided datasets (override with env vars)
RESUME_CSV = os.environ.get("RESUME_CSV", os.path.join(HERE, "..", "Resume", "Resume.csv"))
POSTINGS_CSV = os.environ.get("POSTINGS_CSV", os.path.join(HERE, "..", "..", "postings.csv"))
random.seed(0)

CAT_RE = {
    "ACCOUNTANT": r"\baccountant\b",
    "ADVOCATE": r"\b(attorney|lawyer|paralegal|legal counsel|advocate)\b",
    "AGRICULTURE": r"\b(agricultur\w*|farm|agronom\w*|crop)\b",
    "APPAREL": r"\b(apparel|fashion)\b",
    "ARTS": r"\b(artist|museum|art director|arts)\b",
    "AUTOMOBILE": r"\b(automotive|auto technician|mechanic)\b",
    "AVIATION": r"\b(aviation|aircraft|pilot|airline)\b",
    "BANKING": r"\b(bank teller|teller|banker|loan officer)\b",
    "BPO": r"\b(call center|customer service representative)\b",
    "BUSINESS-DEVELOPMENT": r"\bbusiness development\b",
    "CHEF": r"\b(chef|line cook|sous chef)\b",
    "CONSTRUCTION": r"\b(construction|site superintendent)\b",
    "CONSULTANT": r"\bconsultant\b",
    "DESIGNER": r"\b(graphic designer|interior designer|designer)\b",
    "DIGITAL-MEDIA": r"\b(digital media|social media|digital marketing)\b",
    "ENGINEERING": r"\b(mechanical|electrical|civil|manufacturing) engineer\b",
    "FINANCE": r"\b(financial analyst|finance manager|fp&a)\b",
    "FITNESS": r"\b(fitness|personal trainer)\b",
    "HEALTHCARE": r"\b(registered nurse|medical assistant|healthcare|clinical)\b",
    "HR": r"\b(human resources|hr generalist|hr manager|recruiter)\b",
    "INFORMATION-TECHNOLOGY": r"\b(it support|it specialist|systems administrator|network administrator|help desk)\b",
    "PUBLIC-RELATIONS": r"\b(public relations|communications manager|communications specialist)\b",
    "SALES": r"\b(sales representative|sales associate|account executive)\b",
    "TEACHER": r"\bteacher\b",
}
CLUSTER = {
    "fin": ["ACCOUNTANT", "FINANCE", "BANKING"],
    "biz": ["SALES", "BUSINESS-DEVELOPMENT", "BPO", "CONSULTANT", "HR", "PUBLIC-RELATIONS", "DIGITAL-MEDIA"],
    "creative": ["DESIGNER", "ARTS", "APPAREL", "DIGITAL-MEDIA", "PUBLIC-RELATIONS"],
    "tech": ["ENGINEERING", "INFORMATION-TECHNOLOGY", "CONSTRUCTION", "AUTOMOBILE", "AVIATION"],
}

def related(a, b):
    return a == b or any(a in v and b in v for v in CLUSTER.values())

NAMES = {
    ("White", "F"): ["Emily", "Anne", "Jill", "Allison", "Laurie", "Sarah", "Meredith", "Carrie", "Kristen"],
    ("White", "M"): ["Todd", "Neil", "Geoffrey", "Brett", "Brendan", "Greg", "Matthew", "Jay", "Brad"],
    ("Black", "F"): ["Aisha", "Keisha", "Tamika", "Lakisha", "Tanisha", "Latoya", "Kenya", "Latonya", "Ebony"],
    ("Black", "M"): ["Rasheed", "Tremayne", "Kareem", "Darnell", "Tyrone", "Hakim", "Jamal", "Leroy", "Jermaine"],
}
SURNAMES = {"White": ["Baker", "Kelly", "McCarthy", "Murphy", "Sullivan", "Walsh"],
            "Black": ["Washington", "Jefferson", "Booker", "Banks", "Jackson", "Robinson"]}
# Proxy stress test: name redacted, but a race-correlated affiliation remains.
PROXY = {"White": "Volunteer mentor, Student Union Leadership Program",
         "Black": "Volunteer mentor, Black Student Union Leadership Program"}


def clean(t, n):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t[:n].rsplit(" ", 1)[0]


def main(per_cat=14, jobs_per_cat=4):
    res = pd.read_csv(RESUME_CSV)
    res = res[res.Resume_str.str.len() > 800]
    post = pd.read_csv(POSTINGS_CSV, usecols=["job_id", "title", "description"]).dropna()
    post = post[post.description.str.len() > 400]
    jobs = {}
    for cat, rx in CAT_RE.items():
        m = post[post.title.str.contains(rx, case=False, regex=True)]
        jobs[cat] = m.sample(min(jobs_per_cat, len(m)), random_state=1)
        print(f"{cat:24s} postings matched={len(m):6d}")
    rows = []
    for cat, g in res.groupby("Category"):
        g = g.sample(min(per_cat, len(g)), random_state=2)
        for i, (_, r) in enumerate(g.iterrows()):
            split = "train" if i % 2 == 0 else "test"
            gender = random.choice("MF")
            names = {race: f"{random.choice(NAMES[(race, gender)])} {random.choice(SURNAMES[race])}"
                     for race in ("White", "Black")}
            other = random.choice([c for c in CAT_RE if not related(c, cat)])
            for qual, jc in ((1, cat), (0, other)):
                j = jobs[jc].sample(1, random_state=int(r.ID) % 10007).iloc[0]
                rows.append(dict(resume_id=int(r.ID), category=cat, job_cat=jc, job_id=int(j.job_id),
                                 job_title=j.title, job_desc=clean(j.description, 700),
                                 resume=clean(r.Resume_str, 1500), qualified=qual, gender=gender,
                                 name_White=names["White"], name_Black=names["Black"], split=split))
    df = pd.DataFrame(rows)
    df.to_json(os.path.join(HERE, "pairs.jsonl"), orient="records", lines=True)
    print(df.shape, df.split.value_counts().to_dict(), df.qualified.mean())


if __name__ == "__main__":
    main()
