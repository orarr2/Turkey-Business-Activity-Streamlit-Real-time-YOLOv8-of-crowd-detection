# GCP Billing Kill-Switch

Auto-disable billing on the `turkey-footfall` project the moment a Cloud
Billing budget threshold is crossed. This is the difference between "you got
an email at 3 AM that you're over budget" (a plain budget alert) and "the
services stopped billing you three minutes after crossing $5" (this).

## What lives here

| File              | Purpose |
|-------------------|---------|
| `main.py`         | The Cloud Function. Reads the Pub/Sub payload; if `costAmount >= budgetAmount`, detaches the project from its billing account. |
| `requirements.txt`| Python deps. |
| `README.md`       | You are here. |

## Prerequisites (once)

1. **Enable APIs** (GCP Console → APIs & Services → Enable APIs):
   - `Cloud Pub/Sub API`
   - `Cloud Functions API`
   - `Cloud Build API`
   - `Cloud Billing API`
2. **Create a Pub/Sub topic** the budget will publish to:
   ```
   gcloud pubsub topics create budget-alerts --project=turkey-footfall
   ```
3. **Wire the topic into your budget**: GCP Console → Billing → Budgets &
   alerts → open your `turkey-footfall-safety-net` budget → Manage
   notifications → tick `Connect a Pub/Sub topic to this budget` → pick
   `projects/turkey-footfall/topics/budget-alerts` → Save.
4. **Create the runtime service account for the function**:
   ```
   gcloud iam service-accounts create billing-killswitch \
     --display-name "Billing kill-switch runtime" \
     --project=turkey-footfall
   ```
5. **Grant it two project-level roles**:
   - `roles/billing.projectManager` - has `deleteBillingAssignment`, which is
     what actually unlinks the project from a billing account.
   - `roles/browser` - has `resourcemanager.projects.get`, which the function
     needs to call `projects.getBillingInfo` (the idempotency check that runs
     before the unlink). Without it the function fails with a 403 on that
     read step and never reaches the unlink.

   Note: `roles/billing.projectManager` is a *project-level* role - GCP
   rejects it if you try to attach it to the billing account directly
   (`Role roles/billing.projectManager is not supported for this resource.`).

   Easiest way, from Cloud Shell:
   ```
   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/billing.projectManager

   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/browser
   ```
   Or, via UI: GCP Console → IAM & Admin → IAM (with the project
   `turkey-footfall` selected) → Grant Access:
   - Principal: `billing-killswitch@turkey-footfall.iam.gserviceaccount.com`
   - Roles: **Project Billing Manager** and **Browser**

6. **Force-create the Pub/Sub service agent** (skip only if you've been
   using Pub/Sub push-subscriptions in this project before). GCP creates
   service agents lazily, and the first `gcloud functions deploy` will
   ask to bind `roles/iam.serviceAccountTokenCreator` to the Pub/Sub
   service agent - which fails with `Service account ... does not exist`
   if the agent has not been provisioned yet. Force it now:
   ```
   gcloud beta services identity create --service=pubsub.googleapis.com \
     --project=turkey-footfall
   ```

## Deploy the function

From this folder (`src/deploy/gcp-billing-killswitch/`) on a machine that has
`gcloud` authenticated as an account with Cloud Functions Admin permissions
on the project:

```bash
gcloud functions deploy billing-killswitch \
    --gen2 \
    --project=turkey-footfall \
    --region=us-east1 \
    --runtime=python312 \
    --source=. \
    --entry-point=stop_billing \
    --trigger-topic=budget-alerts \
    --set-env-vars=PROJECT_ID=turkey-footfall \
    --service-account=billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
    --memory=256Mi \
    --timeout=60s \
    --max-instances=1
```

Deploy takes 2-4 minutes. When it's done:
```bash
gcloud functions describe billing-killswitch --gen2 --region=us-east1
```
should print `state: ACTIVE`.

## Grant the trigger SA permission to invoke the function

Cloud Functions gen2 runs on top of Cloud Run. The Eventarc/Pub/Sub trigger
authenticates to the underlying Cloud Run service with an OIDC token whose
subject is the trigger's service account (the one we passed to
`--service-account` at deploy time). That SA needs `roles/run.invoker`
on the Cloud Run service backing the function - otherwise every Pub/Sub
delivery is rejected with:
```
The request was not authenticated. The IAM principal lacks {run.routes.invoke} permission.
```
and the function never runs.

Grant it once, in Cloud Shell:
```
gcloud functions add-invoker-policy-binding billing-killswitch \
  --gen2 \
  --region=us-east1 \
  --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com
```

## Prove it works (recommended)

Temporarily lower the budget threshold below current spend, or publish a
synthetic message to the topic:

```bash
gcloud pubsub topics publish budget-alerts \
    --message='{"budgetDisplayName":"test","costAmount":999,"budgetAmount":1}'
```

Then inspect the function's log:
```bash
gcloud functions logs read billing-killswitch --gen2 --region=us-east1 --limit=20
```

You should see `billing DISABLED on turkey-footfall`, and in the Billing
console the project should show `Billing account: None`.

**Re-enable billing after your test**: GCP Console → Billing → Link this
project to a billing account → pick `My Billing Account`.

## What it does NOT do

- It does not delete resources. The VM, Firestore data, Storage bucket, and
  the function itself all remain intact - they simply stop generating billable
  events until you re-link the billing account.
- It does not touch the free-tier services. The e2-micro VM is Always Free
  and keeps running; the collector process on it continues to try Firestore
  writes and will start returning 429 as usage exceeds free-tier quotas.
- It does not care *which* threshold crossed - Google publishes at every
  configured threshold (50/90/100/120%). The function itself only detaches
  billing when `costAmount >= budgetAmount`. If you want a different rule
  (e.g. only at 120%) tighten the condition in `main.py`.

## Cost of the kill-switch itself

- Pub/Sub topic: one message per threshold cross, essentially free.
- Cloud Function invocation: 2 million invocations/month free tier.
- Cloud Storage for the function source: tiny, within Always Free.

Zero, in practice.
