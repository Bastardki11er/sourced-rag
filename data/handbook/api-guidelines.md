# Public API Guidelines

These rules apply to every endpoint exposed to customers through the public API.

## Versioning

The API version is part of the path, for example `/v2/robots`. We introduce a new major version only for breaking changes. A deprecated version keeps working for at least 12 months after its replacement is announced, and customers receive email reminders 6 months and 1 month before shutdown.

## Resource naming

Use plural nouns for collections (`/v2/tasks`) and nest only one level deep (`/v2/sites/{site_id}/robots`). Field names use snake_case. Timestamps are ISO 8601 strings in UTC.

## Pagination

List endpoints use cursor-based pagination, not page numbers. Responses include a `next_cursor` field, which is null on the last page. The default page size is 50 and the maximum is 200.

## Errors

Errors return a JSON body with `code`, `message`, and `request_id` fields. Use 400 for validation errors, 404 for missing resources, 409 for state conflicts, and 429 when a client exceeds its rate limit. Never return a 200 status with an error in the body.

## Rate limits

Each API token may make up to 600 requests per minute. Responses include `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers. Customers on the Enterprise plan can request a higher limit through their account manager.

## Idempotency

Endpoints that create resources accept an `Idempotency-Key` header. A repeated request with the same key within 24 hours returns the original response instead of creating a duplicate.
