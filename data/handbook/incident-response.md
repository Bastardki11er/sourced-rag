# Incident Response

This page describes how we run an incident from the first page to the postmortem. The on-call rotation itself is described in the On-Call Handbook.

## Roles

Every SEV1 and SEV2 incident has an **incident commander (IC)**. The IC coordinates the response and makes decisions; they should not be the person typing commands to fix the problem. The IC appoints a **communications lead**, who posts updates for customers and internal stakeholders, and one or more **responders**, who investigate and mitigate.

Any engineer who has completed the incident commander training can act as IC. The training is a two-hour workshop offered every month.

## Communication cadence

For a SEV1, the communications lead posts a status update every 30 minutes, even if nothing has changed. For a SEV2, updates go out every 60 minutes. Customer-facing updates are published on the status page; internal updates go to the #incidents channel.

Never speculate about root cause in customer-facing updates. Describe the impact and what we are doing about it.

## Mitigation first

Restore service before you investigate. Rolling back the most recent deploy is the default first step whenever an incident starts within two hours of a release. Feature flags should be turned off before attempting a code fix.

## Postmortems

A written postmortem is required for every SEV1 and SEV2 incident. The draft is due within 5 business days of the incident being resolved, and it is reviewed in the weekly reliability meeting on Thursdays.

Postmortems are blameless: they describe what happened and why the system allowed it, not who made a mistake. Each postmortem must list action items with an owner and a due date. Action items from SEV1 postmortems must be completed within 30 days.

## Closing an incident

The IC declares the incident resolved once the impact has ended and monitoring has been stable for at least 30 minutes.
