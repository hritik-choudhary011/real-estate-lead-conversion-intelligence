# GitHub par upload karne ka guide (Hinglish)

## Option A - Browser se (sabse aasan, Git install nahi chahiye)
1. https://github.com par login karo → upar **+ → New repository**.
2. Repository name: `PropLead-AI` | Public select karo | **README/.gitignore/license add MAT karo** (humare paas already hain) → **Create repository**.
3. Zip file ko apne computer par **Extract** karo.
4. GitHub repo page par **"uploading an existing file"** link par click karo.
5. Extract kiye hue `PropLead-AI` folder ke **andar ki saari files aur folders** drag-and-drop karo (folder khud nahi, uske andar ka content).
   - Ek baar me 100 files tak upload hoti hain. Agar zyada ho to 2 baar me karo.
6. Neeche commit message likho: `Initial commit - PropLead AI project` → **Commit changes**.

## Option B - Git command line se
```bash
cd PropLead-AI
git init
git add .
git commit -m "Initial commit - PropLead AI project"
git branch -M main
git remote add origin https://github.com/<your-username>/PropLead-AI.git
git push -u origin main
```
(Pehle GitHub par khali repo bana lo, upar Option A ke step 1-2.)

## Upload se PEHLE zaroor karo
- [ ] `README.md` me `<Your Name>`, `<Your Institute>`, `<your-username>`, contact details badlo.
- [ ] `LICENSE` me `<YOUR NAME>` badlo.
- [ ] Koi **real customer data** (naam, phone, email, address) repo me nahi hai - check karo.
- [ ] Dashboard chalakar screenshots lo (`streamlit run app.py`) aur `images/` me daalkar README me jodo.

## Upload ke BAAD
- Repo ke **About** section me description likho: *Real estate lead conversion intelligence - lead scoring, ML model, Streamlit dashboard*.
- Topics add karo: `real-estate`, `lead-scoring`, `machine-learning`, `streamlit`, `data-analytics`, `python`.
- (Optional) Streamlit Community Cloud par free deploy karo: share.streamlit.io → repo select karo → main file `app.py`.
