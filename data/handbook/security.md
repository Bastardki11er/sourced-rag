# Security Practices

Our robots work next to people, so a security problem can become a safety problem. Every engineer is responsible for following these practices.

## Secrets management

Store all secrets, including API keys, database passwords, and signing keys, in Vault. Never commit secrets to a repository or put them in `.env` files that are shared over chat. Services read secrets from Vault at startup using their workload identity.

Credentials must be rotated at least every 90 days. Vault rotates database credentials automatically; API keys for third-party services must be rotated by the owning team.

If you accidentally commit a secret, treat it as compromised: rotate it immediately, then tell the security team. Deleting the commit is not enough.

## Laptop and device security

Company laptops must have full-disk encryption enabled and must lock after 5 minutes of inactivity. Install operating system updates within 7 days of release. Personal devices may be used for email and chat only, never for source code.

## Access reviews

Managers review their team's access to production systems every quarter. Access that has not been used for 60 days is removed automatically. Production database access is granted for a maximum of 8 hours at a time through the access portal.

## Reporting a security issue

Report suspected phishing, lost devices, or vulnerabilities to security@kestrel.example or in the #security-help channel. Lost or stolen laptops must be reported within 1 hour so they can be wiped remotely.

## Robot firmware signing

All firmware deployed to robots must be signed with the production signing key, which is held in a hardware security module. Unsigned firmware is rejected by the robot bootloader.
