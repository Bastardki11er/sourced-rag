# Deploying to Production

We ship the fleet software through scheduled release trains. This page covers the schedule, the canary process, freezes, and how to roll back.

## Release trains

Release trains leave twice a week, on Tuesday and Thursday at 13:00 Pacific Time. A change must be merged to `main` and pass the full integration suite by 11:00 on the day of the train to be included. Changes that miss the cutoff wait for the next train.

Hotfixes for SEV1 or SEV2 incidents can ship outside the train schedule with approval from the incident commander.

## Canary rollout

Every release starts as a canary on 5% of customer sites for 30 minutes. The deploy tool watches three signals during the canary: robot error rate, route-planning latency, and emergency-stop events. If any signal regresses by more than 10% compared with the previous release, the canary is halted automatically.

After a healthy canary, the release goes to 25% of sites, then to 100% after another hour.

## Deploy freezes

There is a company-wide deploy freeze from December 15 to January 2, because most customers run their peak season then. During a freeze, only hotfixes approved by a director can ship. Teams may also request a short freeze for a specific customer launch through the release channel.

## Rolling back

To roll back, run:

```
kctl rollback --to <release-id>
```

Rollbacks do not need approval and should be your first move when a release causes an incident. A rollback takes about 8 minutes to reach every site. After rolling back, open a ticket so the change can be fixed before the next train.

## Feature flags

New behavior that affects robot motion must ship behind a feature flag and be enabled gradually. Flags are managed in the Flagpole dashboard. Remove a flag within 60 days of it reaching 100% so that stale flags do not accumulate.
