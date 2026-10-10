# On-Call Handbook

Every engineer on a product team joins the on-call rotation after their first 90 days. This page explains how the rotation works, what each severity level means, and what we expect from the person holding the pager.

## Rotation schedule

The rotation is weekly. Handoff happens every Tuesday at 10:00 Pacific Time, so that nobody starts a shift on a Monday after a weekend of accumulated alerts. Each team runs a primary and a secondary on-call. The secondary is the outgoing primary from the previous week, which keeps context fresh.

Swaps are allowed at any time as long as both people update the schedule in the paging system before the shift starts. You do not need manager approval for a swap.

## Severity levels

We classify incidents into four levels. When in doubt, pick the higher severity; downgrading later is cheap.

- **SEV1** – A customer-facing outage or a safety risk on the warehouse floor (for example, robots not stopping when an emergency stop is pressed). Acknowledge within 5 minutes, and an incident commander must be assigned within 15 minutes.
- **SEV2** – Major degradation for many customers, such as route planning taking more than 10 seconds. Acknowledge within 15 minutes.
- **SEV3** – A single customer or a non-critical feature is affected. Acknowledge within 1 business hour.
- **SEV4** – Cosmetic issues and internal tooling problems. Handle during normal working hours.

## Escalation

If the primary does not acknowledge a page within the window for its severity, the paging system escalates to the secondary automatically. If the secondary also misses it, the page goes to the team's engineering manager. For SEV1 incidents, the on-call engineer should also page the Fleet Safety team directly, regardless of which component is failing.

## Compensation and time off

On-call engineers receive a stipend of $400 for each week they serve as primary and $150 for each week as secondary. If you are paged between 22:00 and 07:00 local time, you may take the following morning off without using PTO.

## Handoff checklist

At handoff, the outgoing primary writes a short summary in the #oncall-handoff channel covering open incidents, noisy alerts that should be tuned, and any deploys scheduled during the coming week.
