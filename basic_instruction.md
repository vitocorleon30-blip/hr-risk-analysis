Here is a comprehensive project summary in Markdown format, synthesizing the source documentation and the specific technical work we have completed (data structure, logic, and archetypes).

---

#Project Specification: Predictive Attrition Risk Modelling##1. Executive Summary**Goal:** Build a functional AI prototype that predicts which employees are at high risk of resigning before they actually quit.
**Core Value:** Shift HR from "Reactive" (exit interviews) to "Proactive" (early intervention), saving roughly $25M annually for large firms by retaining top talent.
**Approach:** "Vibe Coding" — rapidly building a functional prototype using AI tools to generate code and synthetic data.

---

##2. Team Structure & ResponsibilitiesThe project is divided into three distinct roles:

* **Coders (2 People):**
* 
*Coder A (Data & Model):* Responsible for the backend logic and generating the "rich" dataset.


* 
*Coder B (UI & Prompts):* Responsible for the frontend (Streamlit/Gradio) and designing the "Key Prompts" that explain the risk.




* 
**Documenter (1 Person):** Translates the technical workflow into a business report, focusing on "Problem Identification" and "Implementation".


* 
**Presenter (1 Person):** Visualizes the story and sells the business impact using the "Real-World Impact Example".



---

##3. The Data Foundation ("The Single Source of Truth")To support the "Multi-Modal Input" requirement (structured data + unstructured text) without complex databases, we designed a **Hybrid CSV Architecture**.

###3.1. Dataset Overview* **Format:** Single flat CSV file (`employee_attrition_dataset_final.csv`).
* **Scale:** 500 Unique Employees.
* **Hierarchy:** 6 Levels (CEO, C-Level, Senior Manager, Manager, Standard, Junior).

###3.2. Key Metrics TrackedWe track data across four distinct dimensions to ensure the model finds "Root Causes":

| Category | Metric | Type | Purpose |
| --- | --- | --- | --- |
| **Real-Time Behavior** | `Daily_Log_JSON` | JSON Array (60 floats) | Visualizes work habits (hours worked) day-by-day for the last 2 months.

 |
| **Longitudinal Trends** | `Performance_History_JSON` | JSON Array (4 ints) | Tracks Q1→Q4 performance to detect declines. |
|  | `Engagement_History_JSON` | JSON Array (4 ints) | Tracks Q1→Q4 sentiment to spot "Quiet Quitting". |
| **Health & Burnout** | `Sick_Days_60D` | Integer | Recent health spikes indicating stress. |
|  | `Vacation_Days_12M` | Integer | Low usage (<5 days) flags high burnout risk. |
| **Qualitative Context** | `Manager_Notes` | Text (30-50 words) | Detailed observations for NLP analysis (e.g., "Visibly exhausted"). |
|  | `Survey_Comments` | Text (30-50 words) | The employee's direct voice ("I feel stuck"). |

---

##4. The "Story-Driven" Logic (Archetypes)To ensure the demo is compelling and strictly avoids "oxymorons" (e.g., high performance data paired with "poor results" text), we implemented a logic engine that assigns "At Risk" employees (10% of workforce) to one of three specific archetypes.

###4.1. Archetype A: The "Burnout" Case* **The Story:** High performer working unsustainable hours who is physically crashing.
* **Data Signature:**
* **Performance:** High (90+) but slightly dipping in Q4.
* **Hours:** Extreme (> 9.5 avg).
* **Vacation:** **0-2 days** (Major Red Flag).
* **Sick Days:** High spike (5-15 days) in last 60 days.
* **Text:** "Results are excellent, but they look visibly exhausted."



###4.2. Archetype B: The "Quiet Quitter"* **The Story:** An employee who has mentally checked out and is doing the bare minimum.
* **Data Signature:**
* **Engagement:** Steep decline (`[7, 5, 3, 2]`).
* **Hours:** Minimum threshold (exactly 7.0 - 7.5).
* **Responsiveness:** Peers report "Hard to reach on Slack."
* **Text:** "Doing the bare minimum to get by."



###4.3. Archetype C: The "Stalled Career"* **The Story:** A capable employee frustrated by lack of growth.
* **Data Signature:**
* **Tenure:** High (> 4 years).
* **Last Promotion:** > 36 months ago.
* **Performance:** Stable/Good (No drop in quality).
* **Engagement:** Consistently Low (3-5/10).
* **Text:** "Reliable performer, but expressed frustration about lack of promotion."



---

##5. Technical Architecture & Workflow###5.1. The Pipeline1. **Data Ingestion:** The Python script generates unique employees and assigns archetypes to ensure valid correlations.
2. **Pattern Recognition (Backend):**
* Uses **Random Forest Classifier** logic to analyze the inputs.


* Features used for training: `Avg_Daily_Hours`, `Sick_Days`, `Engagement_Score` (Current), `Performance_Trend` (Slope).


3. **Explainability (XAI):**
* The model does not just output a score. It cross-references the `Archetype` logic to generate a natural language explanation (e.g., *"Risk is High due to declining engagement and zero vacation usage"*).




4. **Visualization (Frontend):**
* **Coder B** builds a Streamlit app that parses the `JSON` columns to render line charts for "Work Habits" and "Performance Trends."



###5.2. Technology Stack* 
**Core:** Python (pandas, numpy, json).


* 
**ML Libraries:** scikit-learn (Random Forest), SHAP (for explainability).


* 
**Interface:** Streamlit or Gradio.



---

##6. Strategic & Ethical ConsiderationsThe project documentation emphasizes that this tool must be used responsibly.

* **Transparency:** Avoid "Black Box" predictions. The UI must explain *why* someone is flagged (using the `Manager_Notes` and variable weights).


* **Ethics:** Data must be anonymized (mock IDs used in demo). The tool is for *support*, not punishment. Managers are trained to use these insights for "Retention Plans," not firing.


* 
**Bias Check:** The model must be audited to ensure it doesn't disproportionately flag specific demographics.



##7. Next Steps for Implementation1. **Coder A:** Run the final `generate_final_dataset.py` script to produce the CSV.
2. **Coder B:** Ingest the CSV into Streamlit. Parse `Daily_Log_JSON` to create the interactive "Activity Graph."
3. **Presenter:** Use the "Burnout" Archetype as the primary example in the "Demo" slide to show the tool catching a high-performer before they quit.