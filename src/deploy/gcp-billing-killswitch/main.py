"""Hard kill-switch for GCP billing.

Triggered by a Pub/Sub message from a Cloud Billing budget alert (Google
Cloud emits one every time a budget threshold is crossed). Reads the
current cost and budget from the payload; if `cost >= budget` it detaches
the project from its billing account. All billable services stop within
minutes:

    * The e2-micro VM is Always Free and keeps running, but any request
      it makes to Firestore or Storage above the free-tier quota starts
      returning 429 - so the collector fails cleanly instead of costing.
    * Firebase Console shows "billing disabled" on the project.

To re-enable billing after the alert clears you have to manually re-link
the project to a billing account in the GCP Console.

Deployment: see README.md in this directory.

References:
  https://cloud.google.com/billing/docs/how-to/notify#cap_disable_billing_to_stop_usage
"""
from __future__ import annotations

import base64
import json
import os

from googleapiclient import discovery
import functions_framework  # provided by the Cloud Functions Python runtime


PROJECT_ID = os.environ["PROJECT_ID"]  # set via --set-env-vars at deploy time
PROJECT_NAME = f"projects/{PROJECT_ID}"


@functions_framework.cloud_event
def stop_billing(cloud_event) -> None:
    """Cloud Functions gen2 entry point.

    `cloud_event.data` is the CloudEvents envelope; the Pub/Sub body is a
    base64-encoded JSON string under `.message.data`.
    """
    envelope = cloud_event.data or {}
    encoded = (envelope.get("message") or {}).get("data")
    if not encoded:
        print("no message.data in envelope - nothing to do")
        return

    payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
    cost   = float(payload.get("costAmount",   0))
    budget = float(payload.get("budgetAmount", 0))
    name   = payload.get("budgetDisplayName", "(unnamed budget)")
    print(f"budget={name!r}  cost={cost}  limit={budget}")

    if budget <= 0:
        print("budget amount is zero or missing - not acting")
        return
    if cost < budget:
        print(f"under budget ({cost} < {budget}) - not acting")
        return

    billing = discovery.build("cloudbilling", "v1", cache_discovery=False)
    proj = billing.projects().getBillingInfo(name=PROJECT_NAME).execute()
    if not proj.get("billingEnabled"):
        print("billing already disabled - noop")
        return

    print(f"DISABLING billing on {PROJECT_ID} (cost={cost} >= budget={budget})")
    billing.projects().updateBillingInfo(
        name=PROJECT_NAME,
        body={"billingAccountName": ""},
    ).execute()
    print(f"billing DISABLED on {PROJECT_ID}")
