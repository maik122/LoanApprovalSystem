# LoanIQ — Product Roadmap

Prioritised using the **MoSCoW** framework. Items within each tier are ordered by impact.

---

## Must Have
*The system is not production-ready without these.*

### Security & Authentication
- [ ] Add user authentication (OAuth2 / username+password) — right now anyone with the URL can access the app and run predictions
- [ ] Role-based access control: `analyst` (read-only Explorer + Model tabs) vs `officer` (full Predict access) vs `admin` (can retrain)
- [ ] Audit log — every prediction made should be recorded with timestamp, user, inputs, and output

### Input Validation
- [ ] Reject or warn on nonsensical inputs (e.g. loan amount > 10× annual income, CIBIL score of exactly 300 or 900)
- [ ] Enforce required fields on the prediction form — currently the form submits even with zeroed-out values
- [ ] Sanitise all string inputs before they reach the model

### Prediction Logging & Traceability
- [ ] Persist every prediction to a database (SQLite locally, PostgreSQL in production) with full input snapshot
- [ ] Each prediction gets a unique reference ID that can be cited in a loan file
- [ ] Make the prediction explainable — show which features pushed the decision (SHAP values)

### Model Versioning
- [ ] Version every saved model with a timestamp and dataset hash — `models/v1_20250315/`
- [ ] Track which model version produced which prediction
- [ ] Prevent the app from silently running a stale model after the dataset changes

### Error Handling
- [ ] Graceful fallback UI when model files are missing — currently shows a plain red text block
- [ ] Catch and display encoder/model mismatch errors with a clear "please retrain" message
- [ ] Handle malformed or missing CSV columns without a full stack trace

---

## Should Have
*Significantly improves usability and trust. Should ship in v1.1.*

### Explainability
- [ ] SHAP waterfall chart on the Predict page — show each feature's contribution to the final decision
- [ ] Natural language explanation: *"This application was rejected primarily because the CIBIL score of 520 is below the threshold typically associated with approval."*
- [ ] Add a "borderline cases" indicator when confidence is between 45–55% — flag for mandatory human review

### Retraining Workflow Inside the App
- [ ] Admin page to upload a new dataset CSV and trigger retraining without touching the terminal
- [ ] Show a diff of key metrics (F1, accuracy, feature importances) between current and new model before promoting
- [ ] One-click model promotion with automatic rollback if new model is worse

### Batch Prediction
- [ ] Upload a CSV of multiple applicants and get back a results file
- [ ] Progress bar for large batches
- [ ] Downloadable output CSV with prediction, confidence score, and reference ID per row

### Data Quality Dashboard
- [ ] Show data drift alerts — flag when the distribution of incoming prediction inputs starts diverging from the training data
- [ ] Missing value report on uploaded datasets before training
- [ ] Duplicate applicant detection (same loan_id or identical feature set submitted twice)

### UI Improvements
- [ ] Mobile-responsive layout — the current grid breaks on screens below 768px
- [ ] Dark/light mode toggle
- [ ] Loading skeleton screens instead of spinner while charts render
- [ ] Keyboard navigation support for the prediction form

---

## Could Have
*Nice to have. Adds polish and depth but not blocking.*

### Advanced Analytics
- [ ] Cohort analysis — approval rates segmented by time period, region, or applicant group
- [ ] Model calibration plot — check if a predicted 70% confidence actually corresponds to 70% real-world approval rate
- [ ] Compare two model versions side-by-side on the Model tab

### Alternative Models
- [ ] Add XGBoost and Random Forest to the training grid and include them in the comparison
- [ ] Add an ensemble (soft-voting) option that combines all tuned models
- [ ] Threshold tuning UI — let the user adjust the classification threshold and see how precision/recall trade off in real time

### Notifications
- [ ] Email or Slack notification when a batch job finishes
- [ ] Alert when model performance on recent predictions drops below a configurable F1 threshold
- [ ] Weekly summary report auto-generated and emailed to admins

### Developer Experience
- [ ] Dockerfile + docker-compose so the app spins up in one command anywhere
- [ ] GitHub Actions CI pipeline — run tests and linting on every push
- [ ] Pre-commit hooks for code formatting (black, ruff)
- [ ] Unit tests for the preprocessing pipeline and prediction logic

### Export & Integrations
- [ ] Export individual prediction results as a formatted PDF report
- [ ] REST API endpoint (FastAPI) so external systems can call the model programmatically
- [ ] Webhook support to push prediction results to a CRM or loan origination system

---

## Won't Have
*Explicitly out of scope for this project. These belong in a different product.*

- **Bureau API integration** — pulling live CIBIL/Experian scores via API at prediction time. Requires licensed bureau access, compliance agreements, and a separate data infrastructure layer. Out of scope here.
- **Automated loan approval without human sign-off** — this tool is a decision-support system, not a decisioning engine. Final approval must always have a human in the loop given regulatory requirements (GDPR Article 22, RBI guidelines).
- **Custom model architecture** — deep learning, transformer-based credit scoring, or graph neural networks for relationship-based fraud detection. The current tree/logistic baseline is appropriate for the dataset size and interpretability requirements.
- **Multi-tenancy** — serving multiple separate banks or lending institutions from a single deployment with data isolation. Requires a fundamentally different infrastructure design.
- **Real-time streaming ingestion** — consuming applications from Kafka or a message queue. This is an offline batch tool.
- **Mobile native app** — iOS/Android wrapper. Not the right interface for a loan officer workflow tool.

---

## Version Targets

| Version | Focus | Key Deliverables |
|---------|-------|-----------------|
| `v1.0` | ✅ Current | Core prediction, Explorer, Model tab, dark UI |
| `v1.1` | Hardening | Auth, audit log, input validation, SHAP explainability |
| `v1.2` | Workflow | Batch prediction, in-app retraining, model versioning |
| `v2.0` | Platform | REST API, Docker, data drift monitoring, admin dashboard |